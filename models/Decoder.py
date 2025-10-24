import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder1(nn.Module):
    def __init__(self, channel):
        super(Decoder1, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.convout = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, f1, f2, f3, f4=None, f5=None):
        fg = F.interpolate(f3, size=f1.size()[2:], mode='bilinear')
        f2 = self.conv1(fg * f2) + f2
        f1 = self.conv2(fg * f1) + f1
        out = self.convout(torch.cat((f2, f1), 1))

        return out
class Decoder2(nn.Module):
    def __init__(self, channel):
        super(Decoder2, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.convout = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
    def forward(self, f1, f2, f3, f4):
        fg = F.interpolate(f4, size=f1.size()[2:], mode='bilinear')
        fg2 = F.interpolate(f3, size=f1.size()[2:], mode='bilinear')
        f2 = self.conv1(fg * fg2 * f2) + f2
        f1 = self.conv2(fg * fg2 * f1) + f1
        out = self.convout(torch.cat((f2, f1), 1))

        return out


class Decoder3(nn.Module):
    def __init__(self, channel):
        super(Decoder3, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.convout = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, f1, f2, f3, f4, f5):
        fg = F.interpolate(f5, size=f1.size()[2:], mode='bilinear')
        fg2 = F.interpolate(f4, size=f1.size()[2:], mode='bilinear')
        fg3 = F.interpolate(f3, size=f1.size()[2:], mode='bilinear')
        f2 = self.conv1(fg * fg2 * fg3 * f2) + f2
        f1 = self.conv2(fg * fg2 * fg3 * f1) + f1
        out = self.convout(torch.cat((f2, f1), 1))

        return out