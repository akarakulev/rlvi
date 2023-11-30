import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms

    
class LeNet(nn.Module):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
    ])

    def __init__(self, input_channel=3, num_classes=10, k=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6 * k, kernel_size=5)
        self.conv2 = nn.Conv2d(6 * k, 16 * k, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNetDO(nn.Module):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
    ])

    def __init__(self, input_channel=3, num_classes=10, k=1, dropout_rate=0.25):
        super(LeNetDO, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6 * k, kernel_size=5)
        self.conv2 = nn.Conv2d(6 * k, 16 * k, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout_rate = dropout_rate


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out, p=self.dropout_rate)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out, p=self.dropout_rate)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
