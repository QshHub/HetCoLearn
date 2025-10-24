import torch
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    def __init__(self):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(12, 6, 1)
    def forward(self, x1, x2):
        fused_features = torch.cat((x1, x2), dim=1)
        x = self.conv1(fused_features)

        return x
if __name__ == '__main__':
    rgb1 = torch.randn(4, 3, 256, 256).cuda()
    rgb2 = torch.randn(4, 3, 256, 256).cuda()

    net = FeatureFusionModule().cuda()
    outs = net(rgb1,rgb2)
    for out in outs:
        print(out.shape)