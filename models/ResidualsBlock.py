import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np
from toolbox.models.TestNet.models.BasicConv import BasicConv2d

class Residuals(nn.Module):
    def __init__(self, channel):
        super(Residuals, self).__init__()

        self.residuals_conv1_0 = BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        self.residuals_conv1 = BasicConv2d(channel, channel, 1)
        self.residuals_conv1_1 = BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1))
        self.residuals_conv1_2 = BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        self.residuals_conv2_0 = BasicConv2d(channel, channel, 3, padding=5, dilation=5)
        self.residuals_conv2 = BasicConv2d(channel, channel, 1)
        self.residuals_conv2_1 = BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2))
        self.residuals_conv2_2 = BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        self.conv1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.convsc = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
    def forward(self,x):
        x_r_1 = self.residuals_conv1_0(x)
        x_r_2 = self.residuals_conv1(x)
        x_r_2 = self.residuals_conv1_1(x_r_2)
        x_r_2 = self.residuals_conv1_2(x_r_2)
        x_out = self.conv1(torch.cat([x_r_1, x_r_2], 1)) + x
        x_r_1 = self.residuals_conv2_0(x_out)
        x_r_2 = self.residuals_conv2(x_out)
        x_r_2 = self.residuals_conv2_1(x_r_2)
        x_r_2 = self.residuals_conv2_2(x_r_2)
        x_out2 = self.conv2(torch.cat([x_r_1, x_r_2], 1)) + x_out + x
        x_casa = torch.mul(self.sa(self.ca(x_out2)),self.ca(x_out2))
        out = self.convsc(x_casa) + x_out2 + x_out + x

        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)