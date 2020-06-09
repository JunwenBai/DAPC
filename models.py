import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seed=19940423
np.random.seed(seed)
torch.manual_seed(seed)

class DNN(nn.Module):
    
    def __init__(self, n_input, n_output, n_hid=16, use_sigmoid=False):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_output)
        self.lists = nn.ModuleList([self.fc1, self.fc2, self.fc3])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lists:
            stdv = 1. / math.sqrt(layer.weight.size(1))
            stdv = 0.5
            layer.weight.data.normal_(stdv)
            if layer.bias is not None:
                layer.bias.data.normal_(stdv)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x



class Match_DNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(Match_DNN, self).__init__()
        self.fc = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x
