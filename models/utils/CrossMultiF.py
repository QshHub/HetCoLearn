import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusionModule(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, output_channels):
        super(CrossModalFusionModule, self).__init__()

        # 用于将CNN输出的特征图调整到和Transformer的空间尺寸一致
        self.cnn_projection = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=1)

        # 用于调整CNN和Transformer输出特征图的通道数一致
        self.channel_projection = nn.Conv2d(cnn_channels + transformer_channels, output_channels, kernel_size=1)

        # 用于计算跨模态注意力
        self.attention = nn.MultiheadAttention(embed_dim=output_channels, num_heads=8)

        # 用于归一化
        self.batch_norm = nn.BatchNorm2d(output_channels)

        # 用于激活
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cnn_features, transformer_features):
        # 1. 对CNN特征图进行1x1卷积以调整通道数
        cnn_features = self.cnn_projection(cnn_features) # Shape: [B, C_CNN, H, W]

        # 2. 对Transformer特征图进行上采样以匹配CNN的空间尺寸
        _, _, H_cnn, W_cnn = cnn_features.shape
        transformer_features = F.interpolate(transformer_features, size=(H_cnn, W_cnn), mode='bilinear', align_corners=False)

        # 3. 融合CNN和Transformer的特征图
        fused_features = torch.cat((cnn_features, transformer_features), dim=1) # Shape: [B, C_CNN + C_Transformer, H, W]

        # 4. 对融合后的特征图进行通道调整
        fused_features = self.channel_projection(fused_features) # Shape: [B, C_output, H, W]

        # 5. 对融合特征进行BatchNorm和激活
        fused_features = self.batch_norm(fused_features)
        fused_features = self.relu(fused_features)

        # 6. 使用注意力机制增强特征表示
        # 变换为 [B * H * W, C] 的形状以适应Attention的输入格式
        B, C, H, W = fused_features.shape
        fused_features = fused_features.view(B * H * W, C).unsqueeze(1) # Shape: [B * H * W, 1, C]

        # 计算注意力权重
        attention_output, _ = self.attention(fused_features, fused_features, fused_features) # Shape: [B * H * W, 1, C]
        # 还原形状为 [B, C, H, W]
        attention_output = attention_output.view(B, H, W, C).permute(0, 3, 1, 2) # Shape: [B, C, H, W]
        return attention_output