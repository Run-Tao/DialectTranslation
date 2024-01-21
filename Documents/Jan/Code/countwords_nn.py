import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 80)
        self.fc2 = nn.Linear(80, 64)
        self.fc3 = nn.Linear(64, 15)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
