import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn as nn
import argparse
import os, sys
import numpy as np
import copy
import subprocess
from vgg16 import VGG16_BN
sys.path.append('..')
import argparse
from load_dataset import load_cifar

def test_early_exit(model,test_loader,train_on_gpu,number_of_branch=3):
    model.eval()
    correct_final = 0
    correct_classifiers = np.zeros((number_of_branch))
    total = 0

    # Iterate through the test dataset
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # Forward pass through the model
            outputs = model(inputs)[0]

            for i in range(len(outputs)):
                _, predicted = torch.max(outputs[i], 1)
                correct_classifiers[i]+= (predicted == labels).sum().item()
            
            total += labels.size(0)

    print("Test Accuracy:")
    for i in range(len(correct_classifiers)):
        xx=correct_classifiers[i]
        correct_classifiers[i]=100 * correct_classifiers[i] / total
        print(f"    Classifier {i}: {correct_classifiers[i]}% , correct: {xx}, total samples {total}")

    return correct_classifiers

def train_early_exit(model,optimizer_m,epoch,criterion,train_loader,test_loader,train_on_gpu,number_of_branches,losses_coefficient):
    print('\nEpoch: %d' % epoch)
    model.train()

    correct_classifiers = np.zeros((number_of_branches))
    total = 0

    global best_acc
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer_m.zero_grad()
        outputs = model(inputs)[0]
        if not(type(outputs) is tuple):
            outputs=[outputs]
    
        loss = criterion(outputs[0], targets) * losses_coefficient[0]
        for i in range (1,len(outputs)):
             loss += criterion(outputs[i], targets) * losses_coefficient[i]

        loss.backward()
        optimizer_m.step()

        for i in range(len(outputs)):
            _, predicted = torch.max(outputs[i], 1)
            correct_classifiers[i]+=(predicted == targets).sum().item()

        total += targets.size(0)


    # Print training accuracy for each classifier at the end of each epoch
    print(f"- Training Accuracy: Epoch {epoch + 1} ")
    for i in range(len(correct_classifiers)):
            correct_classifiers[i]= 100 * correct_classifiers[i] / total
            print(f"    Classifier {i}: {correct_classifiers[i]}%")
    return correct_classifiers

if __name__ == "__main__":
    from vgg16 import VGG16_BN
    from load_dataset import load_cifar
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_exits', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=70)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--normal_data_path', type=str, default="~/datasets/")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda_device', type=str, default="1")
    parser.add_argument('--losses_coefficient', nargs='+', type=float, 
                        help="List of loss coefficients for each exit, e.g. --losses_coefficient 5.0 0.3 0.2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    train_loader, test_loader, train_data, test_data = load_cifar(
        args.batch_size, args.num_classes, data_path=args.normal_data_path, num_workers=args.num_workers
    )

    model = VGG16_BN(args.num_classes).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Set losses_coefficient from command line or default to 1.0 for each exit
    if args.losses_coefficient is not None:
        losses_coefficient = args.losses_coefficient
        if len(losses_coefficient) != args.num_exits:
            raise ValueError(f"Number of loss coefficients ({len(losses_coefficient)}) must match num_exits ({args.num_exits})")
    else:
        losses_coefficient = [0.5, 0.3, 0.2]  

    best_acc = 0
    for epoch in range(args.epochs):
        train_acc = train_early_exit(
            model, optimizer, epoch, criterion, train_loader, test_loader, True, args.num_exits, losses_coefficient
        )
        test_acc = test_early_exit(model, test_loader, True, number_of_branch=args.num_exits)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc} | Test Acc: {test_acc} | LR: {scheduler.get_last_lr()[0]:.5f}")

    torch.save(model.state_dict(), "checkpoints/vgg16_early_exit_trained.pth")
    print("Training complete. Model saved as vgg16_early_exit_trained.pth")
