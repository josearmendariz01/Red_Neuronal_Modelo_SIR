import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

class ImprovedSIRNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 2)
            # nn.Sigmoid() or other activation depending on the Beta and Gamma range
        )

    def forward(self, x):
        return self.layers(x)

##################################################
## representing train data as a Dataset object
##################################################

class SIRDataset(Dataset):
    def __init__(self, S, I, R, beta, gamma):
        
        self.S = torch.tensor(S, dtype=torch.float32)
        self.I = torch.tensor(I, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        s = self.S[idx]
        i = self.I[idx]
        r = self.R[idx]
        b = self.beta[idx]
        g = self.gamma[idx]

        x = torch.cat((s, i, r))  # Concatenating the arrays
        y = torch.tensor([b, g], dtype=torch.float32)
        return x, y
    

class SIRNetwork(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
            #nn.Linear(input_size, 512),
            #nn.ReLU(),
            #nn.Linear(512, 256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Linear(64, 32),
            #nn.ReLU(),
            #nn.Linear(32, 2)
        )

    def forward(self, x):
        return(self.layers(x))
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        residual = self.shortcut(residual)
        out += residual
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNetSIR(nn.Module):
    def __init__(self, input_size):
        super(ResNetSIR, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(input_size, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)