"""
V4 - 多普勒专用判别器
专门判别多普勒效应的真实性，不影响背景
复用V3的多普勒提取逻辑，改进为真正的对抗判别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DopplerRegionExtractor(nn.Module):
    """
    多普勒区域提取器
    从频域中精准定位和提取多普勒十字区域
    """
    def __init__(self):
        super().__init__()
    
    def detect_doppler_cross(self, log_magnitude):
        """
        检测多普勒十字区域
        
        Args:
            log_magnitude: (B, 1, H, W) - 频域对数幅度谱
        
        Returns:
            doppler_mask: (B, 1, H, W) - 多普勒区域掩码
        """
        # 垂直方向能量集中（速度轴）
        vertical_energy = log_magnitude.mean(dim=-1, keepdim=True)  # (B, 1, H, 1)
        vertical_threshold = torch.quantile(vertical_energy.flatten(1), 0.8, dim=1, keepdim=True)
        vertical_threshold = vertical_threshold.view(-1, 1, 1, 1)
        vertical_mask = (vertical_energy > vertical_threshold).float()
        
        # 水平方向能量集中（距离轴）
        horizontal_energy = log_magnitude.mean(dim=-2, keepdim=True)  # (B, 1, 1, W)
        horizontal_threshold = torch.quantile(horizontal_energy.flatten(1), 0.8, dim=1, keepdim=True)
        horizontal_threshold = horizontal_threshold.view(-1, 1, 1, 1)
        horizontal_mask = (horizontal_energy > horizontal_threshold).float()
        
        # 多普勒十字 = 垂直 + 水平
        doppler_mask = vertical_mask + horizontal_mask
        doppler_mask = torch.clamp(doppler_mask, 0, 1)
        
        return doppler_mask
    
    def forward(self, image):
        """
        提取多普勒区域的频域表示
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            doppler_region: (B, 1, H, W) - 多普勒区域（频域）
            doppler_mask: (B, 1, H, W) - 多普勒区域掩码
        """
        # FFT变换到频域
        fft = torch.fft.rfft2(image, norm='ortho')
        magnitude = torch.abs(fft)
        
        # 对数幅度谱
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 检测多普勒十字区域
        doppler_mask = self.detect_doppler_cross(log_magnitude)
        
        # 提取多普勒区域
        doppler_region = doppler_mask * log_magnitude
        
        return doppler_region, doppler_mask


class DopplerFeatureEncoder(nn.Module):
    """
    多普勒特征编码器
    将多普勒区域编码为特征向量
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 第1层
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第2层
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第3层
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第4层
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 全局池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.feature_dim = base_channels * 8
    
    def forward(self, doppler_region):
        """
        Args:
            doppler_region: (B, 1, H, W) - 多普勒区域
        
        Returns:
            features: (B, feature_dim) - 特征向量
        """
        features = self.encoder(doppler_region)
        return features


class DopplerOnlyDiscriminator(nn.Module):
    """
    多普勒专用判别器
    
    核心特点：
    1. 只判别多普勒区域的真实性
    2. 不关注背景和其他特征
    3. 真正的对抗训练（输出真/假概率）
    4. 参数可训练，持续进化
    """
    def __init__(self, base_channels=64, dropout=0.3):
        super().__init__()
        
        # 多普勒区域提取器（不需要训练参数）
        self.region_extractor = DopplerRegionExtractor()
        
        # 特征编码器
        self.feature_encoder = DopplerFeatureEncoder(base_channels)
        
        # 判别器分类头
        feature_dim = self.feature_encoder.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim // 4, 1)  # 输出logit（真/假）
        )
    
    def forward(self, image, return_features=False):
        """
        判别图像中多普勒效应的真实性
        
        Args:
            image: (B, 1, H, W) - 输入图像
            return_features: bool - 是否返回中间特征
        
        Returns:
            logits: (B, 1) - 真假判别logit
            features: (B, feature_dim) - 特征向量（如果return_features=True）
            doppler_mask: (B, 1, H, W) - 多普勒掩码（如果return_features=True）
        """
        # 提取多普勒区域
        doppler_region, doppler_mask = self.region_extractor(image)
        
        # 编码为特征
        features = self.feature_encoder(doppler_region)
        
        # 判别真假
        logits = self.classifier(features)
        
        if return_features:
            return logits, features, doppler_mask
        else:
            return logits


def doppler_adversarial_loss(discriminator, real_images, fake_images, mode='generator'):
    """
    多普勒对抗损失
    
    Args:
        discriminator: DopplerOnlyDiscriminator
        real_images: (B, 1, H, W) - 真实图像
        fake_images: (B, 1, H, W) - 生成图像
        mode: str - 'generator' 或 'discriminator'
    
    Returns:
        loss: scalar
        info: dict - 详细信息
    """
    if mode == 'discriminator':
        # 判别器训练：区分真假
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())  # detach生成器梯度
        
        # 真实图像损失（希望判别为真，标签=1）
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        
        # 生成图像损失（希望判别为假，标签=0）
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        
        # 总损失
        loss = (real_loss + fake_loss) / 2
        
        # 计算准确率
        real_pred = (torch.sigmoid(real_logits) > 0.5).float()
        fake_pred = (torch.sigmoid(fake_logits) < 0.5).float()
        real_acc = real_pred.mean()
        fake_acc = fake_pred.mean()
        
        # 增加：返回样本级别的统计（用于梯度累积时的准确率计算）
        batch_size = real_images.shape[0]
        num_correct_real = real_pred.sum().item()
        num_correct_fake = fake_pred.sum().item()
        
        info = {
            'loss': loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_acc': real_acc.item(),
            'fake_acc': fake_acc.item(),
            'real_score': torch.sigmoid(real_logits).mean().item(),
            'fake_score': torch.sigmoid(fake_logits).mean().item(),
            # 新增：用于跨batch统计
            'num_correct_real': num_correct_real,
            'num_correct_fake': num_correct_fake,
            'num_samples': batch_size
        }
        
    elif mode == 'generator':
        # 生成器训练：欺骗判别器
        fake_logits = discriminator(fake_images)
        
        # 希望判别器判别为真（标签=1）
        loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
        
        info = {
            'loss': loss.item(),
            'fake_score': torch.sigmoid(fake_logits).mean().item()
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return loss, info


def doppler_feature_matching_loss(discriminator, real_images, fake_images):
    """
    多普勒特征匹配损失（辅助损失）
    让生成图像的多普勒特征接近真实图像
    
    Args:
        discriminator: DopplerOnlyDiscriminator
        real_images: (B, 1, H, W)
        fake_images: (B, 1, H, W)
    
    Returns:
        loss: scalar
        info: dict
    """
    # 提取特征
    _, real_features, _ = discriminator(real_images, return_features=True)
    _, fake_features, _ = discriminator(fake_images, return_features=True)
    
    # MSE损失
    loss = F.mse_loss(fake_features, real_features)
    
    info = {
        'loss': loss.item(),
        'feature_distance': loss.item()
    }
    
    return loss, info


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("DopplerOnlyDiscriminator V4 测试")
    print("="*60)
    
    # 创建判别器
    discriminator = DopplerOnlyDiscriminator(base_channels=64)
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512)
    fake_image = torch.randn(batch_size, 1, 512, 512)
    
    # 测试前向传播
    print("\n【前向传播测试】")
    logits = discriminator(real_image)
    print(f"输入形状: {real_image.shape}")
    print(f"输出logits: {logits.shape}")
    print(f"判别概率: {torch.sigmoid(logits).mean().item():.4f}")
    
    # 测试返回特征
    logits, features, mask = discriminator(real_image, return_features=True)
    print(f"\n特征形状: {features.shape}")
    print(f"多普勒掩码形状: {mask.shape}")
    print(f"多普勒区域覆盖率: {mask.mean().item():.4f}")
    
    # 测试判别器训练
    print("\n【判别器训练测试】")
    d_loss, d_info = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='discriminator'
    )
    print(f"判别器损失: {d_loss.item():.6f}")
    print(f"  真实图像准确率: {d_info['real_acc']:.4f}")
    print(f"  生成图像准确率: {d_info['fake_acc']:.4f}")
    
    # 测试生成器训练
    print("\n【生成器训练测试】")
    g_loss, g_info = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='generator'
    )
    print(f"生成器对抗损失: {g_loss.item():.6f}")
    
    # 测试特征匹配
    fm_loss, fm_info = doppler_feature_matching_loss(
        discriminator, real_image, fake_image
    )
    print(f"特征匹配损失: {fm_loss.item():.6f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\n【参数统计】")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n" + "="*60)
