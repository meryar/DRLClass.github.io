import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self,inputs,outputs):
        super(DeepQNetwork,self).__init__()
        self.fc1 = nn.Linear(inputs,16)
        self.fc2 = nn.Linear(16,8)
        self.out = nn.Linear(8,outputs)
    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.out(x)
        return x
        