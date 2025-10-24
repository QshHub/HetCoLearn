import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import complexPyTorch.complexLayers as CPL

from toolbox.backbone.transformer.EANet import norm_layer
from toolbox.models.TestNet.models.utils.ComplexConv import ComplexConv2d

def complex_gelu(input):
    return F.gelu(input.real).type(torch.complex64)+1j*F.gelu(input.imag).type(torch.complex64)

class AFLayer(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.complexconv = ComplexConv2d(dim, H, W)
        self.complexbn = CPL.ComplexBatchNorm2d(dim)
        self.complexrelu = complex_gelu
        self.dwconv = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, 1)
        group = [8, 16, 32, 32, 16, 8]
        dia = [3, 2, 1, 1, 1, 1]

        self.dwconv = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        bias = x
        dtype = x.dtype

        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # shortcut
        x2 = self.conv(x)  # bchw
        complex_x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")  # 傅里叶变换
        complex_x1 = self.complexconv(complex_x)
        complex_x2 = self.complexrelu(self.complexbn(complex_x1))

        complex_x3 = self.complexbn(complex_x1)

        real = (complex_x3.real * complex_x2.real - complex_x3.imag * complex_x2.imag)
        imag = (complex_x3.real * complex_x2.imag + complex_x2.real * complex_x3.imag)

        complex_x2 = real.type(complex_x1.dtype) + 1j * imag.type(complex_x1.dtype)

        F_x = torch.fft.irfft2(complex_x2, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.permute(0, 2, 3, 1)  # bhwc
        x = x.permute(0, 3, 1, 2)  # bchw
        x = torch.cat((x, F_x), dim=1)  # channel 拼接 2dim
        x = self.dwconv(x)  # 2dim--dim
        x = x.reshape(B, C, N).permute(0, 2, 1)
        x = x.type(dtype)
        x2 = x2.reshape(B, C, N).permute(0, 2, 1)
        output = F.gelu(x + x2)
        output = output.permute(0, 2, 1).reshape(B, -1, H, W)
        return output

if __name__ == '__main__':
    rgb = torch.randn(4, 1, 32, 32).cuda()
    ndsm = torch.randn(4, 16, 256, 256).cuda()

    net = AFLayer(1,32,32).cuda()
    outs = net(rgb)
    print(outs.shape)