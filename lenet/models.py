import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import torch
from torch import nn
from d2l import torch as d2l

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(3, 6, kernel_size = 5, padding = 2)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 2)
        self.conv_2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 2)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, 7)
    
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv_1(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = self.conv_2(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc_1(x)
        x = self.sigmoid(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        x = self.fc_3(x)

        h = x.view(x.shape[0], -1)
        return x , h




# net = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
#                     nn.AvgPool2d(kernel_size=2, stride=2),
#                     nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
#                     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#                     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
#                     nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))