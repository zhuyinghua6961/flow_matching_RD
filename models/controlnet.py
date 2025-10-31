"""
ControlNet - 控制目标位置和条件注入
基于热力图和仿真RD图作为条件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ResBlock, SelfAttention, SinusoidalTimeEmbedding


class ZeroConv(nn.Module):
    """Zero-initialized卷积层，用于ControlNet输出"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # 初始化为0
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlDownBlock(nn.Module):
    """ControlNet的下采样块（简化版）"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0,
                 num_layers=2, downsample=True, use_attention=False):
        super().__init__()
        
        self.resblocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            ) for i in range(num_layers)
        ])

        self.attention = SelfAttention(out_channels) if use_attention else None
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = None
        
        # Zero-Conv输出
        self.zero_conv = ZeroConv(out_channels, out_channels)

    def forward(self, x, time_emb):
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        # 先下采样
        if self.downsample is not None:
            x = self.downsample(x)
        
        # 下采样后再保存输出（与UNet的特征层级对齐）
        output = self.zero_conv(x)
        
        return x, output


class ControlNet(nn.Module):
    """
    ControlNet分支
    
    输入: concat[sim_RD(3ch), heatmap(1ch)] = 4通道条件
    输出: list of feature maps 注入到UNet的各层
    """
    def __init__(
        self,
        cond_channels=4,  # sim_RD(3) + heatmap(1)
        base_channels=32,  # 比UNet轻量（UNet用64）
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.0,
        time_emb_dim=256
    ):
        super().__init__()
        
        self.cond_channels = cond_channels
        self.base_channels = base_channels
        
        # 时间嵌入（与UNet共享结构）
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        
        # 初始卷积（处理4通道条件输入）
        self.conv_in = nn.Conv2d(cond_channels, base_channels, 3, padding=1)
        self.zero_conv_in = ZeroConv(base_channels, base_channels)
        
        # 计算每层的通道数
        channels = [base_channels * mult for mult in channel_mult]
        
        # Encoder（复制UNet的编码器结构）
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.down_blocks.append(ControlDownBlock(
                in_ch,
                out_ch,
                time_emb_dim,
                dropout,
                num_res_blocks,
                downsample=(i != len(channels) - 1),
                use_attention=(i in attention_levels)
            ))
            in_ch = out_ch
        
        # Middle block
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_emb_dim, dropout)
        self.mid_attn = SelfAttention(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_emb_dim, dropout)
        self.mid_zero_conv = ZeroConv(channels[-1], channels[-1])

    def forward(self, x, time):
        """
        Args:
            x: (B, 4, 512, 512) concat[sim_RD, heatmap]
            time: (B,) 时间步 [0, 1]
        Returns:
            outputs: list of feature maps，用于注入UNet
        """
        # 时间嵌入
        time_emb = self.time_embedding(time)
        
        # 初始卷积
        h = self.conv_in(x)
        h = F.silu(h)
        
        # 保存输出
        outputs = [self.zero_conv_in(h)]
        
        # Encoder
        for down_block in self.down_blocks:
            h, output = down_block(h, time_emb)
            outputs.append(output)
        
        # Middle block
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        outputs.append(self.mid_zero_conv(h))
        
        return outputs


if __name__ == "__main__":
    # 测试
    controlnet = ControlNet(
        cond_channels=4,
        base_channels=32,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3)
    )
    
    cond = torch.randn(2, 4, 512, 512)  # sim_RD(3) + heatmap(1)
    t = torch.rand(2)
    
    outputs = controlnet(cond, t)
    
    print(f"ControlNet输入: {cond.shape}")
    print(f"ControlNet输出层数: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Layer {i}: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.2f}M")

