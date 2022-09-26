import torch
import torchvision
import torch.optim as optim
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class SimpleNet(Module):
    def __init__(self, inp_dim, out_dim, width, num_layers, dropout_p=0, activation=None):
        super(SimpleNet, self).__init__()

        self.dropout_p = dropout_p
        
        if activation is None or activation == "relu":
            self.activation = F.relu
        if activation == "sigmoid":
            self.activation = F.hardsigmoid
        if activation == "tanh":
            self.activation = F.hardtanh

        self.fc_input = nn.Linear(inp_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers - 1)])
        self.fc_final = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.activation(self.fc_input(x))
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        x = self.fc_final(x)
        return x