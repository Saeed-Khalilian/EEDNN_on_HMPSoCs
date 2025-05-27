import torch ,torchvision
from torch import nn
import copy, pickle, numpy as np, sys, time, random
from torchsummary import summary
from torchsummary import summary

class VGG16_BN(nn.Module):

    def fc_layer_size(self,in_channels,in_height,in_width,kernel_size,stride,padding):
        out_height= np.floor( ((in_height-kernel_size + 2 * padding)/stride)+1   ) 
        out_width= np.floor( ((in_width-kernel_size + 2 * padding)/stride)+1   ) 
        return int(out_height * out_width * in_channels)


    def __init__(self, num_classes=10):
        super(VGG16_BN, self).__init__()

        self.features1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )

        self.features1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.features2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features5 = nn.Sequential(    
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.exit1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.classifier1= nn.Linear(8192 , num_classes)
        
        self.exit2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.classifier2= nn.Linear(4096 , num_classes)  


    def forward(self, x):

        x = self.features1_1(x)
        fm1 = self.exit1(x)
        fm1=torch.flatten(fm1, 1)
        x1=self.classifier1(fm1)

        x = self.features1_2(x)


        x = self.features2_1(x)
        fm2=self.exit2(x)
        fm2=torch.flatten(fm2, 1)
        x2=self.classifier2(fm2)
        x = self.features2_2(x)


        x = self.features3(x)

        x = self.features4(x)


        x = self.features5(x)
        x = self.avgpool(x)
        fm = torch.flatten(x, 1)
        x = self.classifier(fm)

        # return (x1, x2, x),  (fm1,fm2)
        return (x,x2, x1),  (fm, fm2,fm1)

# vgg19_bn = VGG16_BN(num_classes=100).cuda()
# summary(vgg19_bn, input_size=(3, 32, 32))

