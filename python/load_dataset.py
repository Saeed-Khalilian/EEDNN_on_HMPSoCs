import torchvision.transforms as transforms
import torch, copy, torchvision
import sys
from autoaugment import CIFAR10Policy
from cutout import Cutout

def load_cifar(batch_size=128,num_classes=100, data_path='~/datasets/', num_workers=2):
    
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    

    if num_classes==100:
        train_data = torchvision.datasets.CIFAR100(data_path,train=True,download=False,transform=transform_train)
    else:
        train_data = torchvision.datasets.CIFAR10(data_path,train=True,download=False,transform=transform_train)
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    
    if num_classes==100:
        test_data = torchvision.datasets.CIFAR100(data_path, train=False,download=False, transform=transform_test)
    else:
        test_data = torchvision.datasets.CIFAR10(data_path, train=False,download=False, transform=transform_test) 

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    return train_loader,test_loader ,train_data, test_data

def load_train_valid_test_cifar(batch_size=128,num_classes=100, data_path='~/datasets/', num_workers=2, train_size=0.8):
    
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    transform_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


    if num_classes==100:
        full_dataset = torchvision.datasets.CIFAR100('~/datasets/', train=True,download=False)
    else:
        full_dataset = torchvision.datasets.CIFAR10('~/datasets/', train=True,download=False)
        
    train_size = int(train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = transform_train
    val_dataset=copy.deepcopy(val_dataset)
    val_dataset.dataset.transform = transform_val
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if num_classes==100:
        test_data = torchvision.datasets.CIFAR100('~/datasets/', train=False,download=False, transform=transform_test)
    else:
        test_data = torchvision.datasets.CIFAR10('~/datasets/', train=False,download=False, transform=transform_test) 
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    
    return train_loader,valid_loader,test_loader

