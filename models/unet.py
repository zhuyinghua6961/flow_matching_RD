"""
UNet架构 - Flow Matching主干网络
用于RD图的sim2real映射任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch_size,) 范围[0, 1]
        Returns:
            embedding: (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeMLP(nn.Module):
    """时间嵌入MLP"""
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim)
        )

    def forward(self, time_emb):
        return self.mlp(time_emb)


class ResBlock(nn.Module):
    """残差块 with GroupNorm and SiLU"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = TimeMLP(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 时间调制
        if self.time_emb_dim is not None and time_emb is not None:
            time_out = self.time_mlp(time_emb)
            h = h + time_out[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """自注意力层（仅在低分辨率使用）"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape到多头
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)

        # 注意力
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        out = attn @ v

        # Reshape回去
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        out = self.proj(out)

        return out + x


class DownBlock(nn.Module):
    """下采样块"""
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

    def forward(self, x, time_emb):
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0,
                 num_layers=2, upsample=True, use_attention=False):
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
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = None

    def forward(self, x, skip, time_emb):
        # 拼接skip connection
        x = torch.cat([x, skip], dim=1)
        
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    """
    Flow Matching UNet主干
    
    输入: x_t (B, 3, 512, 512) + time (B,) + controlnet_outputs (list)
    输出: 速度场 v (B, 3, 512, 512)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),  # 在哪些层使用attention
        dropout=0.0,
        time_emb_dim=256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 计算每层的通道数
        channels = [base_channels * mult for mult in channel_mult]
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.down_blocks.append(DownBlock(
                in_ch,
                out_ch,
                time_emb_dim,
                dropout,
                num_res_blocks,
                downsample=(i != len(channels) - 1),
                use_attention=(i in attention_levels)
            ))
            in_ch = out_ch
        
        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_emb_dim, dropout)
        self.mid_attn = SelfAttention(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_emb_dim, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(channels)):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1] if i < len(channels) - 1 else base_channels
            
            # Skip connection会加倍输入通道
            self.up_blocks.append(UpBlock(
                in_ch * 2,  # *2 for skip connection
                out_ch,
                time_emb_dim,
                dropout,
                num_res_blocks,
                upsample=(i != len(channels) - 1),
                use_attention=(len(channels) - 1 - i in attention_levels)
            ))
        
        # 输出层
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, time, controlnet_outputs=None):
        """
        Args:
            x: (B, 3, 512, 512) 带噪声的图像
            time: (B,) 时间步 [0, 1]
            controlnet_outputs: list of tensors，从ControlNet来的条件
        Returns:
            v: (B, 3, 512, 512) 预测的速度场
        """
        # 时间嵌入
        time_emb = self.time_embedding(time)
        
        # 初始卷积
        h = self.conv_in(x)
        
        # Encoder（保存skip connections）
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h, time_emb)
            
            # 添加ControlNet输出
            if controlnet_outputs is not None and i < len(controlnet_outputs):
                h = h + controlnet_outputs[i]
            
            skips.append(h)
        
        # Bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Decoder（使用skip connections）
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)
        
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


if __name__ == "__main__":
    # 测试
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3)
    )
    
    x = torch.randn(2, 3, 512, 512)
    t = torch.rand(2)
    
    output = model(x, t)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

