import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np
from toolbox.models.TestNet.models.utils.ManhtAttention import RetNetRelPos2d,VisionRetentionChunk
from toolbox.models.TestNet.models.Attention import SpatialAttention,ChannelAttention
from toolbox.models.TestNet.models.BasicConv import BasicConv2d
from torchvision import transforms
class ManBlock(nn.Module):
    def __init__(self, inchannel):
        super(ManBlock, self).__init__()
        #曼哈顿
        self.pos = RetNetRelPos2d(embed_dim=32, num_heads=4, initial_value=1, heads_range=3)
        self.retention = VisionRetentionChunk(embed_dim=32, num_heads=4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.feature_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BasicConv2d(inchannel, inchannel, 3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 1),
            nn.Sigmoid()
        )
        self.ca = ChannelAttention(in_planes=inchannel)
        self.sa = SpatialAttention()
        self.fusion_branch = nn.Sequential(
            nn.Conv2d(2 * inchannel, inchannel, 1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # self.convlayer = nn.Sequential(
        #     BasicConv2d(inchannel, inchannel, 3, padding=1),
        #     nn.Conv2d(inchannel, inchannel, 1),
        #     nn.ReLU()
        # )
        self.convout= nn.Conv2d(inchannel,inchannel,1)
        self.conv1 = nn.Conv2d(inchannel, inchannel, 1)
        self.conv2 = nn.Conv2d(inchannel, inchannel, 1)


    def forward(self, x, f_m, f_g):
        residual = x
        size = x.size()[2:]
        f_g = F.interpolate(f_g, size=size, mode='bilinear', align_corners=True)
        at = x.permute(0, 2, 3, 1)
        # print(at.shape)
        b,h,w,c = at.size()
        rel_pos = self.pos((h, w), chunkwise_recurrent=True)
        at = self.retention(at,rel_pos)
        at = at.permute(0, 3, 1, 2)
        at_r = self.feature_branch(at)
        f_m = self.conv1(at_r * f_m) + f_m
        f_g = self.conv2(at_r * f_g) + f_g

        ff = self.fusion_branch(torch.cat([f_m, f_g], 1))
        at_ca = self.ca(at)
        at_sa = self.sa(at_ca)
        at_casa = at_ca * at_sa
        out = ff + at_casa
        out = self.convout(out)
        out = out + residual
        # x1 = self.convlayer2(torch.mul(self.relu(self.convlayer(self.ca2(at))),at_ca)) + at + x




        return out
if __name__ == '__main__':
    rgb1 = torch.randn(4, 128, 64, 64).cuda()
    rgb2 = torch.randn(4, 32, 256, 256).cuda()

    net = ManBlock(32).cuda()
    outs = net(rgb2)
    for out in outs:
        print(out.shape)