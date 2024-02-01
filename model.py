import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 32, (2,3), 1)
        self.conv5 = nn.Conv2d(32, 16, (1,3), 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16*3*14, 96)
        # self.fc1 = nn.Linear(720, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 12)
        self.fc5 = nn.Linear(12, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.pre_conv= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
#         # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(128, 64, 3, 1)
#         self.conv3 = nn.Conv2d(64, 64, 3, 1)
#         self.conv4 = nn.Conv2d(64, 32, (2,3), 1)
#         self.conv5 = nn.Conv2d(32, 16, (1,3), 1)
#         self.flat = nn.Flatten()
#         # self.fc1 = nn.Linear(16*3*14, 96)
#         self.fc1 = nn.Linear(720, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 3)
      

#     def forward(self, x):
#         x = nn.LeakyReLU()(self.pre_conv(x))
#         x = nn.LeakyReLU()(self.conv1(x))
#         x = nn.LeakyReLU()(self.conv2(x))
#         x = nn.LeakyReLU()(self.conv3(x))
#         x = nn.LeakyReLU()(self.conv4(x))
#         x = nn.LeakyReLU()(self.conv5(x))
#         x = self.flat(x)
#         x = nn.LeakyReLU()(self.fc1(x))
#         x = nn.LeakyReLU()(self.fc2(x))      
#         x = self.fc3(x)
        
#         return x

if __name__ == '__main__':
    dumm = torch.rand(4,3,10,25)
    model = CNN()
    out = model(dumm)
    print(out.shape)

