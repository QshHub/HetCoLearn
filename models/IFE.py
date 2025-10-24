import torch.nn as nn

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace)

    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.act(out)
        return out

class IFE(nn.Module):
    def __init__(self):
        super(IFE, self).__init__()


        self.conv5 = ConvModule(2048, 1024, 3, 1, 1)
        self.conv4 = ConvModule(1024, 512, 3, 1, 1)
        self.conv3 = ConvModule(512, 256, 3, 1, 1)
        self.conv2 = ConvModule(256, 64, 3, 1, 1)
        self.conv1 = ConvModule(64, 3, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, d5, d4, d3, d2, d1):
        l5 = self.up2(self.conv5(d5))
        l4 = self.up2(self.conv4(d5 + d4))
        l3 = self.up2(self.conv3(d4 + d3))
        l2 = self.up2(self.conv2(d3 + d2))
        out = self.up2(self.conv1(d2 + d1))
        return out