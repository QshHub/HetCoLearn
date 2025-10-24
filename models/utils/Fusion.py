from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)


        self.sptial = sptial


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        bias = False
        atmap = []
        input = torch.cat((x,y),1)

        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        atmap.append(x)

        out = x

        # x = F.avg_pool2d(x, self.sptial)
        # # print(x.shape)# 4,6,32,32
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        # out = self.fc2(x)
        # atmap.append(out)

        return out
if __name__ == '__main__':
    rgb = torch.randn(4, 6, 256, 256).cuda()
    # ndsm = torch.randn(4, 1, 256, 256).cuda()

    net = Fusion_module(6,6,256).cuda()
    outs = net(rgb,rgb)
    print(outs.shape)