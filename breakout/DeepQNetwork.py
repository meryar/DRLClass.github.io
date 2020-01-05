import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, inputs, output):
        super(DeepQNetwork, self).__init__()
        self.inputs = inputs
        self.output = output

        self.conv1 = nn.Conv2d(4, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4)
        self.f = nn.Linear(self.feature_size(), 64)
        self.f2 = nn.Linear(64, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.f(x))
        x = self.f2(x)
        return x[0]

    def feature_size(self):
        x = torch.zeros(1, *self.inputs)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)