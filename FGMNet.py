import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np
from toolbox.models.TestNet.models.BasicConv import TransBasicConv2d,BasicConv2d
from toolbox.models.TestNet.models.FE import FE
from toolbox.models.TestNet.models.Decoder import Decoder1,Decoder2,Decoder3
from toolbox.models.TestNet.models.IFE import ConvModule
from toolbox.models.TestNet.Gauss import GaussianAttention
from toolbox.models.TestNet.models.FFusion import FFTFusionModule
from toolbox.models.TestNet.models.ResidualsBlock import Residuals

class FGMNet(nn.Module):
    def __init__(self):
        super(FGMNet, self).__init__()
        self.convt = nn.Conv2d(3, 1,1)

        self.conv_logit = nn.Conv2d(32, 6, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(6)
        # Depth Encoder

        self.DE_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.DE_conv1.weight.data = torch.unsqueeze(torch.mean(models.resnet50(pretrained=True).conv1.weight.data, dim=1),
                                                                 dim=1)
        self.DE_bn1 = models.resnet18(pretrained=True).bn1
        self.DE_relu = models.resnet18(pretrained=True).relu
        self.DE_maxpool = models.resnet18(pretrained=True).maxpool
        self.DE_layer1 = models.resnet18(pretrained=True).layer1
        self.DE_layer2 = models.resnet18(pretrained=True).layer2
        self.DE_layer3 = models.resnet18(pretrained=True).layer3
        self.DE_layer4 = models.resnet18(pretrained=True).layer4

        # RGB Encoder
        self.RE_conv1 = models.resnet18(pretrained=True).conv1
        self.RE_bn1 = models.resnet18(pretrained=True).bn1
        self.RE_relu = models.resnet18(pretrained=True).relu

        self.RE_maxpool = models.resnet18(pretrained=True).maxpool
        self.RE_layer1 = models.resnet18(pretrained=True).layer1

        self.RE_layer2 = models.resnet18(pretrained=True).layer2

        self.RE_layer3 = models.resnet18(pretrained=True).layer3

        self.RE_layer4 = models.resnet18(pretrained=True).layer4

        # Feature Enhance
        self.FE1 = FE(64)
        self.FE2 = FE(64)
        self.FE3 = FE(128)
        self.FE4 = FE(256)
        self.FE5 = FE(512)



        self.FSM1 = FFTFusionModule(64,16)
        self.FSM2 = FFTFusionModule(64, 16)
        self.FSM3 = FFTFusionModule(128, 16)
        self.FSM4 = FFTFusionModule(256, 16)
        self.FSM5 = FFTFusionModule(512, 16)

        #Deconv Layer
        self.trans_conv1 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)


        self.resi1 = Residuals(32)
        self.resi2 = Residuals(32)
        self.resi3 = Residuals(32)
        self.resi4 = Residuals(32)
        self.resi5 = Residuals(32)

        self.gauss1 = GaussianAttention()
        self.gauss2 = GaussianAttention()
        self.gauss3 = GaussianAttention()
        self.gauss4 = GaussianAttention()
        self.gauss5 = GaussianAttention()

        # Change Channels
        self.rd_conv1 = nn.Conv2d(64, 32, 1)
        self.rd_conv2 = nn.Conv2d(64, 32, 1)
        self.rd_conv3 = nn.Conv2d(128, 32, 1)
        self.rd_conv4 = nn.Conv2d(256, 32, 1)
        self.rd_conv5 = nn.Conv2d(512, 32, 1)

        self.rd_conv_logit5 = nn.Conv2d(512, 1, 1)
        self.rd_conv_logit4 = nn.Conv2d(256, 1, 1)
        self.rd_conv_logit3 = nn.Conv2d(128, 1, 1)
        self.rd_conv_logit2 = nn.Conv2d(64, 1, 1)
        self.rd_conv_logit1 = nn.Conv2d(64, 64, 1)


        self.ic_conv_logit = nn.Conv2d(1, 64, 1)
        self.conv1 = nn.Sequential(
            BasicConv2d(2 * 32, 32, 3, padding=1),
            BasicConv2d(32, 32, 1)
        )

        self.decoder1 = Decoder1(32)
        self.decoder2 = Decoder2(32)
        self.decoder3 = Decoder3(32)

        self.conv2 = nn.Sequential(
            BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 6, 1)
        )

        self.outconv_at = nn.Sequential(
            BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 6, 1)
        )
        self.out1conv = nn.Conv2d(32, 6, 1)
        self.out2conv = nn.Conv2d(32, 6, 1)
        self.out3conv = nn.Conv2d(32, 6, 1)
        self.out4conv = nn.Conv2d(32, 6, 1)

        self.trans_conv_logit1 = TransBasicConv2d(1, 512, kernel_size=2, stride=2,
                                                 padding=0, dilation=1, bias=False)
        self.trans_conv_logit2 = TransBasicConv2d(1, 256, kernel_size=2, stride=2,
                                                  padding=0, dilation=1, bias=False)
        self.trans_conv_logit3 = TransBasicConv2d(1, 128, kernel_size=2, stride=2,
                                                  padding=0, dilation=1, bias=False)



        self.ConvBnRelu1 = ConvModule(512, 256, 3, 1, 1)
        self.ConvBnRelu2 = ConvModule(256, 128, 3, 1, 1)
        self.ConvBnRelu3 = ConvModule(128, 64, 3, 1, 1)
        self.ConvBnRelu4 = ConvModule(128, 6, 3, 1, 1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    def forward(self, rgb, ndsm):
        rgb = rgb
        depth = ndsm
        # depth = self.convt(rgb)
        size = rgb.size()[2:]
        #Depth Layer

        depth = self.DE_conv1(depth)
        depth = self.DE_bn1(depth)
        depth = self.DE_relu(depth)
        d1 = depth

        depth = self.DE_maxpool(depth)
        depth = self.DE_layer1(depth)
        d2 = depth

        depth = self.DE_layer2(depth)
        d3 = depth

        depth = self.DE_layer3(depth)
        d4 = depth

        depth = self.DE_layer4(depth)
        d5 = depth

        #Feature Enhance
        d1 = self.FE1(d1)
        d2 = self.FE2(d2)
        d3 = self.FE3(d3)
        d4 = self.FE4(d4)
        d5 = self.FE5(d5)




        #RGB Layer
        rgb = self.RE_conv1(rgb)
        rgb = self.RE_bn1(rgb)
        rgb = self.RE_relu(rgb)
        r1 = rgb
        r1 = self.FE1(r1)
        r1 = self.FSM1(d1,r1)
        rgb = self.RE_maxpool(rgb)
        rgb = self.RE_layer1(rgb)

        r2 = rgb
        r2 = self.FE2(r2)
        r2 = self.FSM2(d2, r2)
        rgb = self.RE_layer2(rgb)

        r3 = rgb
        r3 = self.FE3(r3)
        r3 = self.FSM3(d3, r3)
        rgb = self.RE_layer3(rgb)

        r4 = rgb
        r4 = self.FE4(r4)

        r4 = self.FSM4(d4, r4)

        rgb = self.RE_layer4(rgb)

        r5 = rgb
        r5 = self.FE5(r5)

        r5 = self.FSM5(d5, r5)

        # IFE
        r5_logit = r5
        r5_logit = self.rd_conv_logit5(r5_logit)
        r5_logit = self.trans_conv_logit1(r5_logit)
        r5_logit = self.up2(self.ConvBnRelu1(r5_logit))

        r4_logit = r4
        r4_logit = self.rd_conv_logit4(r4_logit)
        r4_logit = self.trans_conv_logit2(r4_logit) + r5_logit
        r4_logit = self.up2(self.ConvBnRelu2(r4_logit))

        r3_logit = r3
        r3_logit = self.rd_conv_logit3(r3_logit)
        r3_logit = self.trans_conv_logit3(r3_logit) + r4_logit
        r3_logit = self.up2(self.ConvBnRelu3(r3_logit))

        r2_logit = r2
        r1_logit = r1
        r2_logit = self.up2(self.rd_conv_logit1(r2_logit)) + r1_logit
        r2_logit = self.rd_conv_logit2(r2_logit)
        r2_logit = self.ic_conv_logit(r2_logit)

        out_logit = torch.cat([torch.mul(r2_logit,r3_logit),r2_logit],dim=1)
        out_logit = self.up2(self.ConvBnRelu4(out_logit))






        r1 = self.rd_conv1(r1)
        r1 = self.resi1(r1)
        r1 = self.gauss1(r1)

        r2 = self.rd_conv2(r2)
        r2 = self.resi2(r2)
        r2 = self.gauss2(r2)

        r3 = self.rd_conv3(r3)
        r3 = self.resi3(r3)
        r3 = self.gauss3(r3)

        r4 = self.rd_conv4(r4)
        r4 = self.resi4(r4)
        r4 = self.gauss4(r4)

        r5 = self.rd_conv5(r5)
        r5 = self.resi5(r5)
        r5 = self.gauss5(r5)

        out_at = r4 + self.up2(r5)
        out_at = r3 + self.up2(out_at)
        out_at = r2 + self.up2(out_at)
        out_at = r1 + self.up2(out_at)
        out_at = self.up2(self.outconv_at(out_at))

        out1 = self.out1conv(self.up32(r5))
        out = self.conv1(torch.cat([r4, self.trans_conv1(r5)], dim=1))
        out2 = self.out2conv(self.up16(out))
        out = self.decoder1(r3, self.trans_conv2(out), r5)
        out3 = self.out3conv(self.up8(out))
        out = self.decoder2(r2, self.trans_conv3(out), r4, r5)
        out4 = self.out4conv(self.up4(out))
        out = self.decoder3(r1, self.trans_conv4(out), r3, r4, r5)
        out_pred = self.conv2(out)
        # print(out_pred.shape)
        out_pred = F.interpolate(out_pred, size=size, mode='bilinear', align_corners=True)
        # print(out_pred.shape)

        return out_pred, out_logit, out_at, out1, out2, out3, out4
if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256).cuda()
    ndsm = torch.randn(4, 1, 256, 256).cuda()

    net = FGMNet().cuda()
    outs = net(rgb,ndsm)
    for out in outs:
        print(out.shape)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True,
                                              verbose=True)
    print(flops)
    print(params)
