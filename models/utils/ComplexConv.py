import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import complexPyTorch.complexLayers as CPL


class BAM(nn.Module):

    def __init__(self, in_channels, W, H, freq_sel_method='top16'):
        super(BAM, self).__init__()
        self.in_channels = in_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.tw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                            groups=self.in_channels)
        self.twln = nn.LayerNorm([self.in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))

    def forward(self, x):
        N, C, H, W = x.shape
        x_s = (self.wmax * self.maxpool(x).squeeze(-1)) + self.wdct * (self.gap(x).squeeze(-1))
        x_s = x_s.unsqueeze(-1)
        att_c = self.sigmoid(self.twln(self.tw(x_s)))
        return att_c

class gcc_dk(nn.Module):

    def __init__(self, channel, direction, W, H):
        super(gcc_dk, self).__init__()
        self.direction = direction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = BAM(channel, W, H)

        self.kernel_generate_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 1), padding=(0, 0), bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), padding=(0, 0), bias=False, groups=channel),
        )

    def forward(self, x):
        glob_info = self.att(x)

        H_info = torch.mean(x, dim=1, keepdim=True)

        H_info = self.kernel_generate_conv(H_info)

        kernel_input = H_info * glob_info.expand_as(H_info)

        return kernel_input
class ComplexConv2d(nn.Module):
    def __init__(self, channels, H, W, dtype=torch.complex64):
        super(ComplexConv2d, self).__init__()
        self.dtype = dtype
        self.conv_r1 = gcc_dk(channels, 'W', W, H // 2 + 1)
        self.conv_i1 = gcc_dk(channels, 'H', W, H // 2 + 1)
        self.conv_i2 = gcc_dk(channels, 'W', W, H // 2 + 1)
        self.conv_r2 = gcc_dk(channels, 'H', W, H // 2 + 1)
        self.weights_h = nn.Parameter(torch.randn(channels, 1, H // 2 + 1))
        self.weights_w = nn.Parameter(torch.randn(channels, W, 1))
        self.H = H // 2 + 1
        self.W = W

    def forward(self, input):
        a = self.conv_r1(input.real)
        B, C, _, _ = a.shape
        b = self.conv_r2(input.real).expand(B, C, self.W, self.H)
        a = a.expand(B, C, self.W, self.H)

        a = self.weights_w * a + self.weights_h * b
        b = self.weights_w * self.conv_r1(input.imag).expand(B, C, self.W, self.H) + self.weights_h * self.conv_r2(
            input.imag).expand(B, C, self.W, self.H)
        c = self.weights_h * self.conv_i1(input.real).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.real).expand(B, C, self.W, self.H)
        d = self.weights_h * self.conv_i1(input.imag).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.imag).expand(B, C, self.W, self.H)

        real = (a - c)
        imag = (b + d)
        return real.type(self.dtype) + 1j * imag.type(self.dtype)