# %% 
import torch, os, sys
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
sys.path.append('pytorch-self-distillation-final')
sys.path.append('vgg16')
from vgg16 import VGG16_BN
import argparse
from resnet import resnet18, resnet50
import torch.nn.functional as F
import pickle,sys, numpy as np , copy
from load_dataset import load_cifar


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--thresholds', nargs='+', type=float, default=[0.9,0.9,0.9,0])
parser.add_argument('--cuda_device', type=str, default="1")
parser.add_argument('--pretrained_model', type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16"])
parser.add_argument('--weights_path', type=str, default="pytorch-self-distillation-final/checkpoints/resnet18.pth")
parser.add_argument('--normal_data_path', type=str, default="~/datasets/")
parser.add_argument('--corrupted_data_path', type=str, default="/data/shared/CIFAR-100-C/")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_exits', type=int, default=4, help='Number of early exit branches (ResNet18 has 4 branches and VGG16 has 3 branches)')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
batch_size = args.batch_size
num_classes = args.num_classes
thresholds = args.thresholds
number_of_branches = args.num_exits

train_loader,test_loader, train_data, test_data=load_cifar(batch_size,num_classes, data_path=args.normal_data_path, num_workers=args.num_workers)

if args.pretrained_model == "resnet18":
    net = resnet18().cuda()          
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.cuda(), labels.cuda()   
    outputs, outputs_feature = net(inputs)
    layer_list = []
    teacher_feature_size = outputs_feature[0].size(1)
    for index in range(1, len(outputs_feature)):
                        student_feature_size = outputs_feature[index].size(1)
                        layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
    net.adaptation_layers = nn.ModuleList(layer_list)
    net.adaptation_layers.cuda()
    net.load_state_dict(torch.load(args.weights_path))

elif args.pretrained_model == "resnet50":
    net = resnet50().cuda()          
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.cuda(), labels.cuda()   
    outputs, outputs_feature = net(inputs)
    layer_list = []
    teacher_feature_size = outputs_feature[0].size(1)
    for index in range(1, len(outputs_feature)):
                        student_feature_size = outputs_feature[index].size(1)
                        layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
    net.adaptation_layers = nn.ModuleList(layer_list)
    net.adaptation_layers.cuda()
    net.load_state_dict(torch.load(args.weights_path))
elif args.pretrained_model == "vgg16":
     net=VGG16_BN(num_classes).cuda()
     net.load_state_dict(torch.load(args.weights_path))

else:
    raise ValueError("Invalid pretrained_model. Choose 'resnet18', 'resnet50' or 'vgg16'.")


# print(test_early_exit(net,test_loader,True,number_of_branch=4))
# print("---------------------------------------------------------------------------")

# ResNet18 has 4 branches
if args.pretrained_model == "resnet18" or args.pretrained_model == "resnet50":
    # thresholds=[1.1,1.1,1.1,0]
    thresholds=[0.9,0.9,0.9,0]
    # thresholds=[0.8,0.8,0.8,0]
    # thresholds=[0.6,0.6,0.6,0]
    # thresholds=[0.4,0.4,0.4,0]

# VGG16 has 3 branches
if args.pretrained_model == "vgg16":
    # thresholds=[1.1,1.1,0]
    thresholds=[0.9,0.9,0]
    # thresholds=[0.8,0.8,0]
    # thresholds=[0.6,0.6,0]
    #thresholds=[0.4,0.4,0]

def exit_policy_confidence(**kwargs ):
    output = kwargs['output']
    threshold = kwargs['threshold']
    labels = kwargs['labels']
    probs = F.softmax(output, dim=0)
    first_max_prob, pred = torch.max(probs, dim=0)
    if first_max_prob > threshold:
        if pred.item() == labels.item():
            return True, True
        else:
            return True, False
    return False, False

def test_early_exit_threshold(model,test_loader,number_of_branches,thresholds):
    model.eval()
    total_samples = 0
 
    classifiers_utility = [0 for _ in range(number_of_branches)]
    correct_predictions = 0
    
    exit_policy= exit_policy_confidence

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs, features = model(images)

            for i in range(len(images)):

                total_samples += 1

                for j in range(len(outputs)-1,-1,-1):
                    exited,classified= exit_policy(output=outputs[j][i], threshold=thresholds[len(outputs)-1-j], labels=labels[i])
                    if exited:
                      classifiers_utility[len(outputs)-1-j] += 1
                      if classified:
                            correct_predictions += 1
                      break
    

    classifiers_utility = np.array(classifiers_utility)
    classifiers_utility_percentage = (classifiers_utility / total_samples) * 100

    print("total samples:", total_samples)
    print("correct predictions:", correct_predictions)
    print('classifiers_utility_percentage', classifiers_utility_percentage)
    print('ACC: ', correct_predictions / total_samples * 100,  "Error:",100-( correct_predictions / total_samples * 100))
    return correct_predictions / total_samples * 100, classifiers_utility

def test_mCE_early_exit_threshold(model,test_loader,number_of_branches,thresholds):

    exit_policy= exit_policy_confidence
    model.eval()

    def show_performance(distortion_name):
        errs = []

        classifiers_utility_percentage = [0 for _ in range(number_of_branches)]

        severity=1
        for severity in range(1, 2):
            
            full_data_pth = os.path.join(args.corrupted_data_path, f"{distortion_name}.npy")
            full_labels_pth = os.path.join(args.corrupted_data_path, "labels.npy")
            test_data.data = np.load(full_data_pth)
            test_data.targets = torch.LongTensor(np.load(full_labels_pth))

            print('Loaded CIFAR-100-C data',full_data_pth,full_labels_pth)
            distorted_dataset_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            distorted_dataset=test_data
            total_samples = 0
 
            classifiers_utility = [0 for _ in range(number_of_branches)]
            correct_predictions = 0

            for batch_idx, (data, labels) in enumerate(distorted_dataset_loader):
                data = data.cuda()

                outputs, features = model(data)

                for i in range(len(labels)):
                        total_samples += 1
                        for j in range(len(outputs)-1,-1,-1):
                            exited,classified= exit_policy(output=outputs[j][i], threshold=thresholds[len(outputs)-1-j], labels=labels[i])
                            if exited:
                              classifiers_utility[len(outputs)-1-j] += 1
                              if classified:
                                    correct_predictions += 1
                              break
            classifiers_utility = np.array(classifiers_utility)
            for i in range(number_of_branches):
                classifiers_utility_percentage[i] +=(classifiers_utility[i] / total_samples) * 100

            print( correct_predictions/ len(distorted_dataset),classifiers_utility)
            errs.append(1 - 1.*correct_predictions / total_samples)

        for i in range(number_of_branches):
            classifiers_utility_percentage[i] = classifiers_utility_percentage[i] / severity
        
        print('Classifiers Utility:',  classifiers_utility_percentage)
        return np.mean(errs ),classifiers_utility_percentage


    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ,'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

    error_rates = []
    classifiers_utility = [0 for _ in range(number_of_branches)]
    count = 0
    for distortion_name in distortions:
        rate, class_u = show_performance(distortion_name)
        error_rates.append(rate)
        count += 1
        print(count,'Distortion: {:15s}| CE (%): {:.2f}'.format(distortion_name, 100 * rate), '\n\n')
        for i in range(number_of_branches):
             classifiers_utility[i] += class_u[i] 

    for i in range(number_of_branches):
             classifiers_utility[i] = classifiers_utility[i]/ len(distortions)

    print('Average mCE(%): {:.2f}'.format(100 * np.mean(error_rates)),' classifiers_utility',classifiers_utility)
    return 100 * np.mean(error_rates)

test_early_exit_threshold(net,test_loader,number_of_branches=number_of_branches,thresholds=thresholds)

# test_mCE_early_exit_threshold(net,None,number_of_branches=number_of_branches,thresholds=thresholds)