import torch
import torch.nn as nn
import pywt
import torch.nn.functional as F
from toolbox.models.TestNet.models.utils.WaveBlock import WaveletConv
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
        return self.sigmoid(out)


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







# 主要模块
class WaveletSegmentationModule(nn.Module):
    def __init__(self, channel):
        super(WaveletSegmentationModule, self).__init__()
        self.waveletconvlayer1 = WaveletConv(channel, wavelet='haar', initialize=True)

        self.waveletconvlayer2 = WaveletConv(channel, wavelet='haar', initialize=True)

        self.waveletconvlayer3 = WaveletConv(channel, wavelet='haar', initialize=True)

        self.convlayer1 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.convlayer2 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        )
        self.convlayer3 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0))
        )
        self.convc1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.convc2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.convc3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv1_1 = BasicConv2d(channel, channel, 1)
        self.conv1_2= BasicConv2d(channel, channel, 3, padding=7, dilation=7)
        self.ca1 = ChannelAttention(channel)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(channel)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(channel)
        self.sa3 = SpatialAttention()
    def forward(self, x):
        x1 = self.waveletconvlayer1(x)
        xB1 = self.sa1(self.ca1(x)*x)*x
        xB1 = self.convlayer1(xB1)
        out1 = self.convc1(torch.cat((x1, xB1), 1))
        x2 = self.waveletconvlayer2(x)
        xB2 = self.sa2(self.ca2(x) * x) * x
        xB2 = self.convlayer2(xB2)
        out2 = self.convc2(torch.cat((x2, xB2), 1))
        x3 = self.waveletconvlayer2(x)
        xB3 = self.sa3(self.ca3(x) * x) * x
        xB3 = self.convlayer3(xB3)
        out3 = self.convc3(torch.cat((x3, xB3), 1))

        out_0 = out1 + out2 + out3
        out_0 = self.conv1_1(out_0)
        x_0 = self.conv1_2(x)
        out = torch.mul(x_0, out_0) + x


        return out










if __name__ == '__main__':
    rgb = torch.randn(4, 32, 256, 256).cuda()


    net = WaveletSegmentationModule(32).cuda()
    outs = net(rgb)
    for out in outs:
        print(out.shape)