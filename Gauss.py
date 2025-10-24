import torch
import torch.nn as nn


class GaussianAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.h_mean = nn.Parameter(torch.tensor(0.5))
        self.h_log_var = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.w_mean = nn.Parameter(torch.tensor(0.5))
        self.w_log_var = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x):
        B, C, H, W = x.shape

        y_coord = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)

        var_h = torch.exp(self.h_log_var)
        var_w = torch.exp(self.w_log_var)

        gauss_h = torch.exp(-(y_coord - self.h_mean).pow(2) / (2 * var_h + 1e-6))

        gauss_w = torch.exp(-(x_coord - self.w_mean).pow(2) / (2 * var_w + 1e-6))

        attention = torch.outer(gauss_h, gauss_w)  # (H, W)
        attention = attention.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        attention = torch.sigmoid(attention)
        return x * attention + x
if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256).cuda()
    ndsm = torch.randn(4, 1, 256, 256).cuda()

    net = GaussianAttention().cuda()
    outs = net(rgb)
    print(outs.shape)
