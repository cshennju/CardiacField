import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results, target, **kwargs):
        d = {}
        d['gray'] = (results - target['gray'])**2

        return d
