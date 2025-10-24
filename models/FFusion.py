import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from toolbox.models.TestNet.models.Attention import SpatialAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialAttention(nn.Module):
    def __init__(self):
        super(FrequencySpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        magnitude = torch.abs(x)
        avg_out = torch.mean(magnitude, dim=1, keepdim=True)
        max_out, _ = torch.max(magnitude, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv1(out)

        attention_map = self.sigmoid(attention_map)

        out = x * attention_map
        return out

class FFTFusionModule(nn.Module):
    def __init__(self, inchannel, SEratio):
        super(FFTFusionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.liner_s = nn.Conv2d(inchannel * 4, inchannel * 4 // SEratio, kernel_size=1)
        self.liner_e = nn.Conv2d(inchannel * 4 // SEratio, inchannel * 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.hswish = nn.Hardswish()
        self.conv = nn.Conv2d(inchannel * 4, inchannel, kernel_size=1)
        self.fuseblock = FFTFuseBlock()

    def forward(self, t, rgb):
        f = self.fuseblock(t, rgb)

        fuse = torch.cat([t, rgb, f], dim=1)

        fusew = self.relu(self.liner_s(self.avg_pool(fuse)))
        fusew = self.hswish(self.liner_e(fusew))

        fuse = torch.mul(fusew, fuse)
        fuse = self.conv(fuse)

        fuse = fuse + rgb
        return fuse

class FFTFuseBlock(nn.Module):
    def __init__(self):
        super(FFTFuseBlock, self).__init__()
        self.sa = FrequencySpatialAttention()
        self.sig = nn.Sigmoid()

    def forward(self, rgb, t):
        fft_rgb = torch.fft.rfft2(rgb)
        fft_t = torch.fft.rfft2(t)

        fft_fuse = torch.mul(fft_rgb, fft_t)

        sa = self.sa(fft_fuse)
        out = torch.mul(fft_fuse, sa)
        out = torch.fft.irfft2(out)
        out = self.sig(out)

        t = torch.mul(t, out)
        rgb = torch.mul(rgb, out)

        out = torch.cat([rgb, t], dim=1)
        return out



if __name__ == '__main__':
    rgb1 = torch.randn(4, 32, 256, 256).cuda()
    rgb2 = torch.randn(4, 32, 256, 256).cuda()

    net = FFTFusionModule(32,16).cuda()
    outs = net(rgb1,rgb2)
    for out in outs:
        print(out.shape)