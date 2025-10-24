import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 4, 1),
        nn.BatchNorm2d(in_channels // 4),
        nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(in_channels // 4),
        nn.ReLU(inplace=True),
        )

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels // 4, n_filters, 1),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x