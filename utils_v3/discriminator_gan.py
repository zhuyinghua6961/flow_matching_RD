"""
改进的GAN判别器 - 真正的对抗训练
支持标准的对抗损失和特征匹配损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DopplerFeatureExtractor(nn.Module):
    """
    多普勒特征提取器（与原版相同）
    在频域中提取多普勒十字特征
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2)
        )
    
    def detect_doppler_cross(self, log_magnitude):
        """检测多普勒十字区域"""
        # 垂直方向能量集中
        vertical_energy = log_magnitude.mean(dim=-1, keepdim=True)
        vertical_threshold = torch.quantile(vertical_energy.flatten(1), 0.8, dim=1, keepdim=True)
        vertical_threshold = vertical_threshold.view(-1, 1, 1, 1)
        vertical_mask = (vertical_energy > vertical_threshold).float()
        
        # 水平方向能量集中
        horizontal_energy = log_magnitude.mean(dim=-2, keepdim=True)
        horizontal_threshold = torch.quantile(horizontal_energy.flatten(1), 0.8, dim=1, keepdim=True)
        horizontal_threshold = horizontal_threshold.view(-1, 1, 1, 1)
        horizontal_mask = (horizontal_energy > horizontal_threshold).float()
        
        # 多普勒十字
        doppler_mask = vertical_mask + horizontal_mask
        doppler_mask = torch.clamp(doppler_mask, 0, 1)
        
        return doppler_mask
    
    def forward(self, image):
        """提取多普勒特征"""
        # FFT变换到频域
        fft = torch.fft.rfft2(image, norm='ortho')
        magnitude = torch.abs(fft)
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 检测多普勒十字区域
        doppler_mask = self.detect_doppler_cross(log_magnitude)
        
        # 提取多普勒区域
        doppler_region = doppler_mask * log_magnitude
        
        # 特征提取
        features = self.feature_net(doppler_region)
        
        return features


class ClutterFeatureExtractor(nn.Module):
    """
    地杂波特征提取器（与原版相同）
    提取中间那条线
    """
    def __init__(self, base_channels=32):
        super().__init__()
        
        # 空间特征提取
        self.spatial_net = nn.Sequential(
            nn.Conv1d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels)
        )
        
        # 频域特征提取
        self.frequency_net = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels)
        )
        
        # 特征融合
        self.fusion = nn.Linear(base_channels * 2, base_channels)
    
    def extract_center_line(self, image):
        """提取中间那条线"""
        B, C, H, W = image.shape
        center_row = H // 2
        
        # 提取中间几行
        start_row = max(0, center_row - 2)
        end_row = min(H, center_row + 3)
        line_region = image[:, :, start_row:end_row, :]
        
        # 对行求平均
        line = line_region.mean(dim=2)
        
        # 1D卷积提取特征
        line_features = self.spatial_net(line)
        
        return line_features
    
    def extract_low_frequency(self, image):
        """提取低频频域特征"""
        # FFT变换
        fft = torch.fft.rfft2(image, norm='ortho')
        magnitude = torch.abs(fft)
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 创建低频掩码
        B, C, H, W = log_magnitude.shape
        center_h, center_w = H // 2, W // 2
        
        y_coords = torch.arange(H, device=log_magnitude.device).float().view(1, 1, H, 1)
        x_coords = torch.arange(W, device=log_magnitude.device).float().view(1, 1, 1, W)
        
        y_dist = torch.abs(y_coords - center_h)
        x_dist = torch.abs(x_coords - center_w)
        
        low_freq_ratio = 0.2
        threshold_h = H * low_freq_ratio
        threshold_w = W * low_freq_ratio
        
        low_freq_mask = ((y_dist < threshold_h) & (x_dist < threshold_w)).float()
        
        # 提取低频区域
        low_freq_region = low_freq_mask * log_magnitude
        
        # 特征提取
        freq_features = self.frequency_net(low_freq_region)
        
        return freq_features
    
    def forward(self, image):
        """提取地杂波特征"""
        spatial_features = self.extract_center_line(image)
        freq_features = self.extract_low_frequency(image)
        
        # 特征融合
        combined = torch.cat([spatial_features, freq_features], dim=1)
        features = self.fusion(combined)
        
        return features


class AdversarialDiscriminator(nn.Module):
    """
    对抗判别器 - 真正的GAN判别
    输出真/假概率，而不只是特征差异
    """
    def __init__(self, doppler_weight=0.75, clutter_weight=0.25):
        super().__init__()
        
        self.doppler_weight = doppler_weight
        self.clutter_weight = clutter_weight
        
        # 1. 多普勒特征提取器
        self.doppler_extractor = DopplerFeatureExtractor(base_channels=64)
        doppler_feature_dim = 64 * 2
        
        # 2. 地杂波特征提取器
        self.clutter_extractor = ClutterFeatureExtractor(base_channels=32)
        clutter_feature_dim = 32
        
        # 3. 融合特征并判别真假
        total_feature_dim = doppler_feature_dim + clutter_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 输出logit（真/假）
        )
    
    def forward(self, image, return_features=False):
        """
        判别图像真假
        
        Args:
            image: (B, 1, H, W) - 输入图像
            return_features: bool - 是否返回中间特征
        
        Returns:
            logits: (B, 1) - 真假logit（未经过sigmoid）
            features: dict - 中间特征（如果return_features=True）
        """
        # 提取多普勒特征
        doppler_features = self.doppler_extractor(image)
        
        # 提取地杂波特征
        clutter_features = self.clutter_extractor(image)
        
        # 融合特征
        combined_features = torch.cat([doppler_features, clutter_features], dim=1)
        
        # 判别真假
        logits = self.classifier(combined_features)
        
        if return_features:
            return logits, {
                'doppler': doppler_features,
                'clutter': clutter_features,
                'combined': combined_features
            }
        else:
            return logits


def adversarial_gan_loss(discriminator, real_images, fake_images, mode='generator'):
    """
    标准GAN对抗损失
    
    Args:
        discriminator: AdversarialDiscriminator
        real_images: (B, 1, H, W) - 真实图像
        fake_images: (B, 1, H, W) - 生成图像
        mode: str - 'generator' 或 'discriminator'
    
    Returns:
        loss: scalar - 对抗损失
        info: dict - 详细信息
    """
    if mode == 'discriminator':
        # 判别器训练：希望判别器能区分真假
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())  # detach生成器梯度
        
        # 真实图像的损失（希望判别为真，标签=1）
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        
        # 生成图像的损失（希望判别为假，标签=0）
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
        
        info = {
            'loss': loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_acc': real_acc.item(),
            'fake_acc': fake_acc.item(),
            'real_score': torch.sigmoid(real_logits).mean().item(),
            'fake_score': torch.sigmoid(fake_logits).mean().item()
        }
        
    elif mode == 'generator':
        # 生成器训练：希望生成的图像被判别为真
        fake_logits = discriminator(fake_images)
        
        # 生成器希望判别器判别为真（标签=1）
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


def feature_matching_loss(discriminator, real_images, fake_images, 
                         doppler_weight=0.75, clutter_weight=0.25):
    """
    特征匹配损失（辅助损失）
    让生成图像的特征接近真实图像的特征
    
    Args:
        discriminator: AdversarialDiscriminator
        real_images: (B, 1, H, W) - 真实图像
        fake_images: (B, 1, H, W) - 生成图像
        doppler_weight: float - 多普勒特征权重
        clutter_weight: float - 地杂波特征权重
    
    Returns:
        loss: scalar - 特征匹配损失
        info: dict - 详细信息
    """
    # 提取特征
    _, real_features = discriminator(real_images, return_features=True)
    _, fake_features = discriminator(fake_images, return_features=True)
    
    # 多普勒特征匹配
    doppler_diff = F.mse_loss(fake_features['doppler'], real_features['doppler'])
    
    # 地杂波特征匹配
    clutter_diff = F.mse_loss(fake_features['clutter'], real_features['clutter'])
    
    # 综合损失
    loss = doppler_weight * doppler_diff + clutter_weight * clutter_diff
    
    info = {
        'loss': loss.item(),
        'doppler_diff': doppler_diff.item(),
        'clutter_diff': clutter_diff.item()
    }
    
    return loss, info


def combined_gan_loss(discriminator, real_images, fake_images, 
                      mode='generator',
                      adversarial_weight=1.0,
                      feature_matching_weight=1.0,
                      doppler_weight=0.75,
                      clutter_weight=0.25):
    """
    组合GAN损失：对抗损失 + 特征匹配损失
    
    Args:
        discriminator: AdversarialDiscriminator
        real_images: (B, 1, H, W)
        fake_images: (B, 1, H, W)
        mode: str - 'generator' 或 'discriminator'
        adversarial_weight: float - 对抗损失权重
        feature_matching_weight: float - 特征匹配损失权重
        doppler_weight: float - 多普勒权重
        clutter_weight: float - 地杂波权重
    
    Returns:
        total_loss: scalar
        info: dict
    """
    # 对抗损失
    adv_loss, adv_info = adversarial_gan_loss(discriminator, real_images, fake_images, mode)
    
    info = {
        'adversarial_loss': adv_loss.item(),
        **{f'adv_{k}': v for k, v in adv_info.items()}
    }
    
    total_loss = adversarial_weight * adv_loss
    
    # 特征匹配损失（只在生成器训练时使用）
    if mode == 'generator' and feature_matching_weight > 0:
        fm_loss, fm_info = feature_matching_loss(
            discriminator, real_images, fake_images,
            doppler_weight, clutter_weight
        )
        total_loss += feature_matching_weight * fm_loss
        info['feature_matching_loss'] = fm_loss.item()
        info.update({f'fm_{k}': v for k, v in fm_info.items()})
    
    info['total_loss'] = total_loss.item()
    
    return total_loss, info


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("AdversarialDiscriminator 测试")
    print("="*60)
    
    # 创建判别器
    discriminator = AdversarialDiscriminator(
        doppler_weight=0.75,
        clutter_weight=0.25
    )
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512)
    fake_image = torch.randn(batch_size, 1, 512, 512)
    
    # 测试判别器前向传播
    logits = discriminator(real_image)
    print(f"\n判别器输出 logits: {logits.shape}")
    print(f"判别概率: {torch.sigmoid(logits).mean().item():.4f}")
    
    # 测试对抗损失（判别器模式）
    print("\n【判别器训练】")
    d_loss, d_info = adversarial_gan_loss(discriminator, real_image, fake_image, mode='discriminator')
    print(f"判别器损失: {d_loss.item():.6f}")
    print(f"  真实图像损失: {d_info['real_loss']:.6f}")
    print(f"  生成图像损失: {d_info['fake_loss']:.6f}")
    print(f"  真实图像准确率: {d_info['real_acc']:.4f}")
    print(f"  生成图像准确率: {d_info['fake_acc']:.4f}")
    
    # 测试对抗损失（生成器模式）
    print("\n【生成器训练】")
    g_loss, g_info = adversarial_gan_loss(discriminator, real_image, fake_image, mode='generator')
    print(f"生成器对抗损失: {g_loss.item():.6f}")
    
    # 测试特征匹配损失
    fm_loss, fm_info = feature_matching_loss(discriminator, real_image, fake_image)
    print(f"特征匹配损失: {fm_loss.item():.6f}")
    print(f"  多普勒差异: {fm_info['doppler_diff']:.6f}")
    print(f"  地杂波差异: {fm_info['clutter_diff']:.6f}")
    
    # 测试组合损失
    print("\n【组合GAN损失】")
    total_loss, total_info = combined_gan_loss(
        discriminator, real_image, fake_image,
        mode='generator',
        adversarial_weight=1.0,
        feature_matching_weight=1.0
    )
    print(f"总损失: {total_loss.item():.6f}")
    print(f"  对抗损失: {total_info['adversarial_loss']:.6f}")
    print(f"  特征匹配损失: {total_info['feature_matching_loss']:.6f}")
    
    print("\n" + "="*60)
