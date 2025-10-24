import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np
from torch import Tensor

from toolbox.models.TestNet.models.FE import FE,FE2
from toolbox.models.TestNet.models.BasicConv import BasicConv2d,BasicConv2d_1
from toolbox.models.TestNet.models.BasicConv import TransBasicConv2d
from toolbox.models.TestNet.models.ManBlock import ManBlock
from toolbox.models.TestNet.models.wav import WaveletSegmentationModule
# from toolbox.backbone.Mobilenetv2 import MobileNet
from toolbox.backbone.mb2.mobilenetv2 import mobilenet_v2
from toolbox.models.AT import SKAttention




class BasicConv2d_reduce(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicConv2d_reduce, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)



class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.convout = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
    def forward(self, f1, f2, f3):
        fg = F.interpolate(f3, size=f1.size()[2:], mode='bilinear')
        f2 = self.conv1(fg*f2) + f2
        f1 = self.conv2(fg*f1) + f1
        out = self.convout(torch.cat((f2, f1), 1))
        return out


class MWMNet(nn.Module):
    def __init__(self):
        super(MWMNet, self).__init__()
        self.MBE = mobilenet_v2(pretrained=True)

        self.WavEx1 = WaveletSegmentationModule(32)
        self.WavEx2 = WaveletSegmentationModule(32)
        self.WavEx3 = WaveletSegmentationModule(32)
        self.WavEx4 = WaveletSegmentationModule(32)
        self.WavEx5 = WaveletSegmentationModule(32)

        self.Matn1 = ManBlock(32)
        self.Matn2 = ManBlock(32)
        self.Matn3 = ManBlock(32)
        self.Matn4 = ManBlock(32)
        self.Matn5 = ManBlock(32)

        self.FE1 = FE(32)
        self.FE2 = FE(32)
        self.FE3 = FE(32)
        self.FE4 = FE(32)
        self.FE5 = FE(32)

        self.trans_conv1 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.outconv = nn.Sequential(
            BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 6, 1)
        )

        self.outconv_logit = nn.Sequential(
            BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 6, 1)
        )
        self.outconv_at = nn.Sequential(
            BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 6, 1)
        )
        self.out1conv = nn.Conv2d(32,6,1)
        self.out2conv = nn.Conv2d(32,6,1)
        self.out3conv = nn.Conv2d(32,6,1)
        self.out4conv = nn.Conv2d(32,6,1)
        self.convinit = nn.Sequential(
            BasicConv2d(2 * 32, 32, 3, padding=1),
            BasicConv2d(32, 32, 1)
        )
        # self.convinit = nn.Conv2d(64,32,1)

        self.catconv1 = nn.Conv2d(64, 32, 1)
        self.catconv2 = nn.Conv2d(64, 32, 1)
        self.catconv3 = nn.Conv2d(64, 32, 1)
        self.catconv4 = nn.Conv2d(64, 32, 1)
        self.catconv5 = nn.Conv2d(64, 32, 1)

        # self.rdconv1 = nn.Conv2d(16, 32,1)
        # self.rdconv2 = nn.Conv2d(24, 32,1)
        # self.rdconv3 = nn.Conv2d(32, 32,1)
        # self.rdconv4 = nn.Conv2d(160, 32,1)
        # self.rdconv5 = nn.Conv2d(320, 32,1)

        self.SE1 = SKAttention(16,reduction=4)
        self.SE2 = SKAttention(24,reduction=4)
        self.SE3 = SKAttention(32,reduction=4)
        self.SE4 = SKAttention(160,reduction=8)
        self.SE5 = SKAttention(320,reduction=8)

        self.rdconv1 = BasicConv2d(16, 32,3,padding = 1)
        self.rdconv2 = BasicConv2d(24, 32,3,padding = 1)
        self.rdconv3 = BasicConv2d(32, 32,3,padding = 1)
        self.rdconv4 = nn.Sequential(
            BasicConv2d(160,64,3,padding = 1),
            BasicConv2d(64,32,3,padding = 1)
        )
        self.rdconv5 = nn.Sequential(
            BasicConv2d(320, 160, 3,padding = 1),
            BasicConv2d(160, 64, 3,padding = 1),
            BasicConv2d(64,32,3,padding = 1)
        )


        self.decoder1 = Decoder(32)
        self.decoder2 = Decoder(32)
        self.decoder3 = Decoder(32)

        self.at1 = FE2(32)
        self.at2 = FE2(32)
        self.at3 = FE2(32)
        self.at4 = FE2(32)
        self.at5 = FE2(32)
    def forward(self,x):
        size = x.size()[2:]
        f1 = self.MBE.features[0:2](x)
        f2 = self.MBE.features[2:4](f1)
        f3 = self.MBE.features[4:7](f2)
        f4 = self.MBE.features[7:17](f3)
        f5 = self.MBE.features[17:18](f4)
        # print(f1.shape, f2.shape, f3.shape, f4.shape, f5.shape)
        f1 = self.SE1(f1)
        f2 = self.SE2(f2)
        f3 = self.SE3(f3)
        f4 = self.SE4(f4)
        f5 = self.SE5(f5)

        f1 = self.rdconv1(f1)
        f2 = self.rdconv2(f2)
        f3 = self.rdconv3(f3)
        f4 = self.rdconv4(f4)
        f5 = self.rdconv5(f5)

        f_g = self.catconv1(torch.cat([self.up2(f5), f4], 1))
        f_g = self.catconv2(torch.cat([f3, self.up2(f_g)], 1))
        f_g = self.catconv3(torch.cat([f2, self.up2(f_g)], 1))
        f_g = self.catconv4(torch.cat([f1, self.up2(f_g)], 1))

        f1 = self.WavEx1(f1)
        f2 = self.WavEx2(f2)
        f3 = self.WavEx3(f3)
        f4 = self.WavEx4(f4)
        f5 = self.WavEx5(f5)

        f1 = self.at1(f1)
        f2 = self.at2(f2)
        f3 = self.at3(f3)
        f4 = self.at4(f4)
        f5 = self.at5(f5)

        out_logit = f4 + self.up2(f5)
        out_logit = f3 + self.up2(out_logit)
        out_logit = f2 + self.up2(out_logit)
        out_logit = f1 + self.up2(out_logit)
        out_logit = self.up2(self.outconv_logit(out_logit))

        f1_m = self.FE1(f1)
        f2_m = self.FE2(f2)
        f3_m = self.FE3(f3)
        f4_m = self.FE4(f4)
        f5_m = self.FE5(f5)
        #
        f1 = self.Matn1(f1, f1_m, f_g)
        f2 = self.Matn2(f2, f2_m, f_g)
        f3 = self.Matn3(f3, f3_m, f_g)
        f4 = self.Matn4(f4, f4_m, f_g)
        f5 = self.Matn5(f5, f5_m, f_g)

        out_at = f4 + self.up2(f5)
        out_at = f3 + self.up2(out_at)
        out_at = f2 + self.up2(out_at)
        out_at = f1 + self.up2(out_at)
        out_at = self.up2(self.outconv_at(out_at))

        out1 = self.out1conv(self.up32(f5))
        f4 = self.convinit(torch.cat([f4, self.trans_conv1(f5)], dim=1))
        out2 = self.out2conv(self.up16(f4))
        f3 = self.decoder1(f3, self.trans_conv2(f4), f5)
        out3 = self.out3conv(self.up8(f3))
        f2 = self.decoder2(f2, self.trans_conv3(f3), f4)
        out4 = self.out4conv(self.up4(f2))
        f1 = self.decoder3(f1, self.trans_conv4(f2), f3)


        out = F.interpolate(f1, size=size, mode='bilinear', align_corners=True)
        out = self.outconv(out)


        return out,out_logit,out_at,out1,out2,out3,out4
        # return out,out

if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256).cuda()


    net = MWMNet().cuda()
    outs = net(rgb)
    for out in outs:
        print(out.shape)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True,
                                              verbose=True)
    print(flops)
    print(params)