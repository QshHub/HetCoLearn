import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np
from toolbox.models.TestNet.models.Attention import SpatialAttention,ChannelAttention

class FE(nn.Module):
    def __init__(self, inchannels):
        super(FE, self).__init__()
        self.ca = ChannelAttention(in_planes=inchannels)
        self.sa = SpatialAttention()

    def forward(self, x):
        caout = self.ca(x)
        sa = self.sa(caout)
        out = torch.mul(caout, sa)
        return out
class FE2(nn.Module):
    def __init__(self, inchannels):
        super(FE2, self).__init__()
        self.ca = ChannelAttention(in_planes=inchannels)
        self.sa = SpatialAttention()
    def forward(self, x):
        caout = self.ca(x)+x
        sa = self.sa(caout)
        out = torch.mul(x, sa)+x
        return out
