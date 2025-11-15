"""
WGAN-GP多普勒专用Critic
基于V4判别器架构，改为WGAN-GP的Critic实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DopplerRegionExtractor(nn.Module):
    """
    多普勒区域提取器（与V4相同）
    通过频域分析定位多普勒效应区域
    """
    def __init__(self, doppler_threshold=0.3):
        super().__init__()
        self.doppler_threshold = doppler_threshold
    
    def forward(self, image):
        """
        提取多普勒区域
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            doppler_region: (B, 1, H, W) - 多普勒区域图像
            doppler_mask: (B, 1, H, W) - 多普勒区域掩码
        """
        B, C, H, W = image.shape
        
        # FFT分析
        fft = torch.fft.fft2(image)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        
        # 高频成分检测（多普勒通常在高频）
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing='ij'
        )
        
        # 距离中心的距离
        distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        high_freq_mask = (distance > min(H, W) * 0.2).float()
        
        # 高频能量
        high_freq_energy = magnitude * high_freq_mask.unsqueeze(0).unsqueeze(0)
        
        # 空间域的高频响应
        high_freq_spatial = torch.fft.ifft2(torch.fft.ifftshift(
            fft_shift * high_freq_mask.unsqueeze(0).unsqueeze(0)
        )).real
        
        # 多普勒掩码：基于高频能量分布
        energy_map = torch.sum(high_freq_energy, dim=(-2, -1), keepdim=True)
        energy_normalized = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min() + 1e-8)
        
        # 自适应阈值
        doppler_mask = (torch.abs(high_freq_spatial) > self.doppler_threshold).float()
        
        # 应用掩码
        doppler_region = image * doppler_mask
        
        return doppler_region, doppler_mask


class DopplerFeatureEncoder(nn.Module):
    """
    多普勒特征编码器（与V4相同）
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            # Block 1: 512x512 -> 256x256
            nn.Sequential(
                nn.Conv2d(1, base_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Block 2: 256x256 -> 128x128
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
                nn.InstanceNorm2d(base_channels*2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Block 3: 128x128 -> 64x64
            nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
                nn.InstanceNorm2d(base_channels*4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Block 4: 64x64 -> 32x32
            nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
                nn.InstanceNorm2d(base_channels*8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Block 5: 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(base_channels*8, base_channels*8, 4, 2, 1),
                nn.InstanceNorm2d(base_channels*8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        
        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * 8
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            features: (B, feature_dim)
        """
        for block in self.conv_blocks:
            x = block(x)
        
        # 全局池化
        x = self.global_pool(x)  # (B, feature_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (B, feature_dim)
        
        return x


class DopplerOnlyCritic(nn.Module):
    """
    WGAN-GP多普勒专用Critic
    
    与V4判别器的区别：
    1. 输出实数评分而非概率
    2. 无sigmoid激活
    3. 支持梯度惩罚
    4. 更稳定的训练
    """
    def __init__(self, base_channels=64, dropout=0.3):
        super().__init__()
        
        # 多普勒区域提取器（不需要训练参数）
        self.region_extractor = DopplerRegionExtractor()
        
        # 特征编码器
        self.feature_encoder = DopplerFeatureEncoder(base_channels)
        
        # Critic评分头（输出实数，无sigmoid）
        feature_dim = self.feature_encoder.feature_dim
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim // 4, 1)  # 输出实数评分（无激活函数）
        )
    
    def forward(self, image, return_features=False):
        """
        评估图像中多普勒效应的"真实性"评分
        
        Args:
            image: (B, 1, H, W) - 输入图像
            return_features: bool - 是否返回中间特征
        
        Returns:
            scores: (B, 1) - Wasserstein距离评分（实数）
            features: (B, feature_dim) - 特征向量（如果return_features=True）
            doppler_mask: (B, 1, H, W) - 多普勒掩码（如果return_features=True）
        """
        # 提取多普勒区域
        doppler_region, doppler_mask = self.region_extractor(image)
        
        # 编码为特征
        features = self.feature_encoder(doppler_region)
        
        # Critic评分（实数，无sigmoid）
        scores = self.critic_head(features)
        
        if return_features:
            return scores, features, doppler_mask
        else:
            return scores


def gradient_penalty(critic, real_images, fake_images, device, lambda_gp=10.0):
    """
    WGAN-GP梯度惩罚
    
    Args:
        critic: DopplerOnlyCritic
        real_images: (B, 1, H, W) - 真实图像
        fake_images: (B, 1, H, W) - 生成图像
        device: torch.device
        lambda_gp: float - 梯度惩罚系数
    
    Returns:
        penalty: scalar - 梯度惩罚损失
    """
    batch_size = real_images.shape[0]
    
    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Critic评分
    scores = critic(interpolated)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 梯度范数
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # 梯度惩罚：(||∇||₂ - 1)²
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty


def doppler_wgan_gp_loss(critic, real_images, fake_images, mode='generator', lambda_gp=10.0):
    """
    WGAN-GP多普勒损失函数
    
    Args:
        critic: DopplerOnlyCritic
        real_images: (B, 1, H, W) - 真实图像
        fake_images: (B, 1, H, W) - 生成图像
        mode: str - 'generator' 或 'critic'
        lambda_gp: float - 梯度惩罚系数
    
    Returns:
        loss: scalar
        info: dict - 详细信息
    """
    device = real_images.device
    
    if mode == 'critic':
        # Critic训练：最大化 E[D(real)] - E[D(fake)]
        real_scores = critic(real_images)
        fake_scores = critic(fake_images.detach())  # detach生成器梯度
        
        # Wasserstein损失
        wasserstein_loss = fake_scores.mean() - real_scores.mean()
        
        # 梯度惩罚
        gp_loss = gradient_penalty(critic, real_images, fake_images, device, lambda_gp)
        
        # 总损失
        loss = wasserstein_loss + gp_loss
        
        # 统计信息
        info = {
            'loss': loss.item(),
            'wasserstein_loss': wasserstein_loss.item(),
            'gp_loss': gp_loss.item(),
            'real_score': real_scores.mean().item(),
            'fake_score': fake_scores.mean().item(),
            'score_gap': (real_scores.mean() - fake_scores.mean()).item(),
            'gradient_penalty': gp_loss.item(),
            'num_samples': real_images.shape[0]
        }
        
    elif mode == 'generator':
        # 生成器训练：最大化 E[D(fake)]
        fake_scores = critic(fake_images)
        
        # 生成器损失（希望critic给高分）
        loss = -fake_scores.mean()
        
        info = {
            'loss': loss.item(),
            'fake_score': fake_scores.mean().item()
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return loss, info


def doppler_feature_matching_loss(critic, real_images, fake_images):
    """
    多普勒特征匹配损失（辅助损失）
    让生成图像的多普勒特征接近真实图像
    
    Args:
        critic: DopplerOnlyCritic
        real_images: (B, 1, H, W)
        fake_images: (B, 1, H, W)
    
    Returns:
        loss: scalar
        info: dict
    """
    # 提取特征
    _, real_features, _ = critic(real_images, return_features=True)
    _, fake_features, _ = critic(fake_images, return_features=True)
    
    # MSE损失
    loss = F.mse_loss(fake_features, real_features)
    
    info = {
        'loss': loss.item(),
        'feature_distance': loss.item()
    }
    
    return loss, info


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 WGAN-GP DopplerOnlyCritic")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建Critic
    critic = DopplerOnlyCritic(base_channels=64).to(device)
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512).to(device)
    fake_image = torch.randn(batch_size, 1, 512, 512).to(device)
    
    print(f"\n✓ Critic创建成功")
    print(f"  设备: {device}")
    
    # 测试前向传播
    scores = critic(real_image)
    print(f"\n前向传播测试:")
    print(f"  输入形状: {real_image.shape}")
    print(f"  输出评分形状: {scores.shape}")
    print(f"  评分范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    # 测试返回特征
    scores, features, mask = critic(real_image, return_features=True)
    print(f"\n返回特征测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  多普勒掩码形状: {mask.shape}")
    print(f"  多普勒区域覆盖率: {mask.mean().item():.4f}")
    
    # 测试WGAN-GP损失
    print(f"\nWGAN-GP损失测试:")
    c_loss, c_info = doppler_wgan_gp_loss(
        critic, real_image, fake_image, mode='critic'
    )
    print(f"  Critic损失: {c_loss.item():.6f}")
    print(f"  Wasserstein损失: {c_info['wasserstein_loss']:.6f}")
    print(f"  梯度惩罚: {c_info['gp_loss']:.6f}")
    print(f"  真实图像评分: {c_info['real_score']:.4f}")
    print(f"  生成图像评分: {c_info['fake_score']:.4f}")
    print(f"  评分差距: {c_info['score_gap']:.4f}")
    
    g_loss, g_info = doppler_wgan_gp_loss(
        critic, real_image, fake_image, mode='generator'
    )
    print(f"  生成器损失: {g_loss.item():.6f}")
    print(f"  生成器评分: {g_info['fake_score']:.4f}")
    
    # 测试特征匹配损失
    fm_loss, fm_info = doppler_feature_matching_loss(
        critic, real_image, fake_image
    )
    print(f"  特征匹配损失: {fm_loss.item():.6f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数占用显存: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\n✓ 所有测试通过！")
    print("="*60)
