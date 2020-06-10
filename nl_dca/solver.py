import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    
    def __init__(self, n_input, n_output, n_hid=128, use_sigmoid=False):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hid)
        #self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_output)
        #self.lists = nn.ModuleList([self.fc1, self.fc2])

    def forward(self, x):
        x = F.dropout(F.elu(self.fc1(x)), p=0.5)
        #x = F.dropout(F.elu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        return x
