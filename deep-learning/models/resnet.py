"""
ResNet: resnet18 and resnet34 implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Linear(nn.Module):
    def __init__(self, in_features, out_features):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
         
    def forward(self, x):
        x = x.mm(self.w)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0):
        super(BasicBlock, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.dropout_rate > 0:
            out = F.dropout2d(out, p=self.dropout_rate)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    def __init__(self, block, num_blocks, input_channel, num_classes, dropout_rate=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_rate=dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(ResNet):
    def __init__(self, input_channel, num_classes):
        super(ResNet18, self).__init__(
            block=BasicBlock, num_blocks=[2,2,2,2], input_channel=input_channel, num_classes=num_classes
        )
    
class ResNet34(ResNet):
    def __init__(self, input_channel, num_classes):
        super(ResNet34, self).__init__(
            block=BasicBlock, num_blocks=[3,4,6,3], input_channel=input_channel, num_classes=num_classes
        )


class ResNet18DO(ResNet):
    def __init__(self, input_channel, num_classes, dropout_rate=0.25):
        super(ResNet18DO, self).__init__(
            block=BasicBlock, num_blocks=[2,2,2,2], input_channel=input_channel, num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    
class ResNet34DO(ResNet):
    def __init__(self, input_channel, num_classes, dropout_rate=0.25):
        super(ResNet34DO, self).__init__(
            block=BasicBlock, num_blocks=[3,4,6,3], input_channel=input_channel, num_classes=num_classes,
            dropout_rate=dropout_rate
        )
