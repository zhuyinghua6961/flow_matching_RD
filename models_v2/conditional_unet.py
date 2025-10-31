"""
ConditionalUNet - 条件去噪网络
基于SimEncoder的特征进行条件控制
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
    """残差块 with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_mlp = TimeMLP(time_emb_dim, out_channels) if time_emb_dim else None
        
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间嵌入
        if self.time_mlp is not None and time_emb is not None:
            time_out = self.time_mlp(time_emb)
            h = h + time_out[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """自注意力层"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        
        # Attention
        scale = head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks=2, 
                 use_attention=False, dropout=0.0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout
            )
            for i in range(num_res_blocks)
        ])
        
        self.attention = SelfAttention(out_channels) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, time_emb, cond_feature=None):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_dim)
            cond_feature: (B, C, H, W) - 来自SimEncoder的条件特征
        """
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        # 注入条件特征（逐元素相加）
        if cond_feature is not None:
            x = x + cond_feature
        
        # 保存skip connection（下采样前）
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks=2,
                 use_attention=False, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        # 第一个ResBlock的输入通道数 = in_channels(上采样后) + in_channels(skip connection, 相同分辨率)
        # 后续ResBlock的输入/输出都是out_channels
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            if i == 0:
                # 第一个块：输入是拼接后的 (in_channels * 2)，输出是out_channels
                self.res_blocks.append(
                    ResBlock(in_channels * 2, out_channels, time_emb_dim, dropout)
                )
            else:
                # 后续块：输入输出都是out_channels
                self.res_blocks.append(
                    ResBlock(out_channels, out_channels, time_emb_dim, dropout)
                )
        
        self.attention = SelfAttention(out_channels) if use_attention else None
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        
        # 调整尺寸以匹配skip connection（处理奇数尺寸）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class ConditionalUNet(nn.Module):
    """
    条件UNet - 基于SimEncoder特征的去噪网络
    
    特点:
    1. 接收时间步t和噪声图x_t
    2. 接收SimEncoder的多尺度特征
    3. 在各层通过逐元素相加注入条件
    4. 输出预测的速度场v
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        time_embed_dim=256,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=2,
        attention_levels=(3,),  # 只在第4层(64x64)用attention
        dropout=0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim
        
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 通道配置
        channels = [base_channels * m for m in channel_mult]
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            use_attn = i in attention_levels
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_embed_dim, num_res_blocks, use_attn, dropout)
            )
        
        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_embed_dim, dropout)
        self.mid_attention = SelfAttention(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_embed_dim, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            in_ch = channels[i]
            out_ch = channels[i - 1]
            use_attn = (i - 1) in attention_levels
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, time_embed_dim, num_res_blocks, use_attn, dropout)
            )
        
        # 输出卷积
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_features):
        """
        Args:
            x: (B, 1, 512, 512) - 噪声图x_t
            t: (B,) - 时间步 [0, 1]
            cond_features: List[(B, C_i, H_i, W_i)] - SimEncoder的多尺度特征
        Returns:
            v: (B, 1, 512, 512) - 预测的速度场
        """
        # 时间嵌入
        time_emb = self.time_embedding(t)  # (B, time_embed_dim)
        
        # 输入卷积
        h = self.conv_in(x)  # (B, 64, 512, 512)
        
        # Encoder
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            # 注入对应层的条件特征
            cond_feat = cond_features[i] if i < len(cond_features) - 1 else None
            h, skip = down_block(h, time_emb, cond_feat)
            skips.append(skip)
        
        # Bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)
        
        # 注入最深层条件特征
        if len(cond_features) > 0:
            h = h + cond_features[-1]
        
        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, time_emb)
        
        # 输出
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


if __name__ == "__main__":
    # 测试
    from sim_encoder import SimEncoder
    
    encoder = SimEncoder(base_channels=64, channel_mult=(1, 2, 4, 8, 16))
    unet = ConditionalUNet(base_channels=64, channel_mult=(1, 2, 4, 8, 16), attention_levels=(3,))
    
    # 模拟输入
    sim_img = torch.randn(2, 1, 512, 512)
    x_t = torch.randn(2, 1, 512, 512)
    t = torch.rand(2)
    
    # 前向传播
    cond_features = encoder(sim_img)
    v = unet(x_t, t, cond_features)
    
    print("ConditionalUNet 测试:")
    print(f"  输入: x_t={x_t.shape}, t={t.shape}")
    print(f"  条件特征: {[f.shape for f in cond_features]}")
    print(f"  输出: v={v.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\n总参数量: {total_params:,}")

