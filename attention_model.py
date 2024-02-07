import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(128, 64, 3, 1)
#         self.conv3 = nn.Conv2d(64, 64, 3, 1)
#         self.conv4 = nn.Conv2d(64, 32, (2,3), 1)
#         self.conv5 = nn.Conv2d(32, 16, (1,3), 1)
#         self.flat = nn.Flatten()
#         self.fc1 = nn.Linear(16*3*14, 96)
#         # self.fc1 = nn.Linear(720, 96)
#         self.fc2 = nn.Linear(96, 48)
#         self.fc3 = nn.Linear(48, 24)
#         self.fc4 = nn.Linear(24, 12)
#         self.fc5 = nn.Linear(12, 3)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.flat(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)

#         return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.attn1 = Attention_block(128,128,64)
        self.attn2 = Attention_block(64,64,32)
        self.attn3 = Attention_block(16,16,8)

        self.pre_conv= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(128, 32, (2,3), 1)
        self.conv5 = nn.Conv2d(32, 16, (1,3), 1)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2*16*3*14, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
      

    def forward(self, x):

        x1 = nn.LeakyReLU()(self.pre_conv(x))
        x2 = nn.LeakyReLU()(self.conv1(x1))

        x2_inter = self.attn1(x2,x2)
        x2 = torch.cat((x2,x2_inter), dim=1)

        x3 = nn.LeakyReLU()(self.conv2(x2))
        x4 = nn.LeakyReLU()(self.conv3(x3))

        x4_inter = self.attn2(x4,x4)
        x4 = torch.cat((x4,x4_inter), dim=1)

        x5 = nn.LeakyReLU()(self.conv4(x4))
        x6 = nn.LeakyReLU()(self.conv5(x5))

        x6_inter = self.attn3(x6,x6)
        x6 = torch.cat((x6,x6_inter), dim=1)

        x_flat = self.flat(x6)  # Assuming 'flat' is a flattening operation
        x7 = nn.LeakyReLU()(self.fc1(x_flat))
        x8 = nn.LeakyReLU()(self.fc2(x7))
        output = self.fc3(x8)  # Final output

        
        return output

if __name__ == '__main__':
    dumm = torch.rand(4,3,10,24)
    model = CNN()
    out = model(dumm)
    print(out.shape)

