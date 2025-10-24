import torch
import torch.nn as nn
import torch.nn.functional as F


class FastCrossAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FastCrossAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 通道注意力部分
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

        # 空间注意力部分
        self.spatial_conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.spatial_conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # 输入x: [b, c, h, w]

        # 1. 通道注意力
        channel_attention = F.relu(self.fc1(x))  # [b, c/reduction, h, w]
        channel_attention = self.fc2(channel_attention)  # [b, c, h, w]
        channel_attention = torch.sigmoid(channel_attention)  # 归一化到(0, 1)

        # 2. 空间注意力
        spatial_attention = F.relu(self.spatial_conv1(x))  # [b, c, h, w]
        spatial_attention = self.spatial_conv2(spatial_attention)  # [b, c, h, w]
        spatial_attention = torch.sigmoid(spatial_attention)  # 归一化到(0, 1)

        # 将通道和空间注意力相结合
        out = x * channel_attention + x * spatial_attention + x # 元素级相乘

        return out


# 测试模块
if __name__ == '__main__':
    b, c, h, w = 4, 64, 128, 128  # 假设输入为4个样本，64个通道，128x128图像
    x = torch.randn(b, c, h, w)  # 随机生成输入

    attention_module = FastCrossAttentionModule(in_channels=c)
    out = attention_module(x)
    print(out.shape)  # 输出的形状应为 [b, c, h, w]
