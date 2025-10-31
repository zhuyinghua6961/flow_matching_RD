"""
SimEncoder - 仿真图特征提取器
提取多尺度特征用于条件注入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块 with GroupNorm and SiLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                dropout=dropout
            )
            for i in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)
        # 下采样前保存特征（用于条件注入）
        feature = x
        x = self.downsample(x)
        return x, feature


class SimEncoder(nn.Module):
    """
    仿真图编码器 - 提取多尺度特征
    
    架构:
        Input: (B, 1, 512, 512)
        ↓ conv_in
        → (B, 64, 512, 512)
        ↓ down1
        → (B, 128, 256, 256)  → feature1
        ↓ down2
        → (B, 256, 128, 128)  → feature2
        ↓ down3
        → (B, 512, 64, 64)    → feature3
        ↓ down4
        → (B, 1024, 32, 32)   → feature4
        
    Returns: 
        [feature1, feature2, feature3, feature4, bottleneck]
        用于在ConditionalUNet的对应层注入
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),  # [64, 128, 256, 512, 1024]
        num_res_blocks=2,
        dropout=0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 下采样块
        channels = [base_channels * m for m in channel_mult]
        self.down_blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, num_res_blocks, dropout)
            )
        
        # Bottleneck
        self.mid_block = ResBlock(channels[-1], channels[-1], dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 512, 512) - 仿真图
        Returns:
            features: List[(B, C_i, H_i, W_i)] - 多尺度特征
        """
        # 初始卷积
        h = self.conv_in(x)  # (B, 64, 512, 512)
        
        # 下采样并收集特征
        features = []
        for down_block in self.down_blocks:
            h, feat = down_block(h)
            features.append(feat)
        
        # Bottleneck
        h = self.mid_block(h)
        features.append(h)  # 最后一层特征
        
        return features


if __name__ == "__main__":
    # 测试
    encoder = SimEncoder(
        in_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=2
    )
    
    x = torch.randn(2, 1, 512, 512)
    features = encoder(x)
    
    print("SimEncoder 输出:")
    for i, feat in enumerate(features):
        print(f"  Feature {i}: {feat.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n总参数量: {total_params:,}")

