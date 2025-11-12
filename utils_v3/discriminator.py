"""
判别器模块 V3
专门针对多普勒效应和地杂波设计的判别器
重点：多普勒效应（主要）
次要：地杂波（中间那条线）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DopplerFeatureExtractor(nn.Module):
    """
    多普勒特征提取器
    在频域中提取多普勒十字特征（主要）
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
        """
        检测多普勒十字区域
        
        Args:
            log_magnitude: (B, 1, H, W) - 对数幅度谱
        
        Returns:
            doppler_mask: (B, 1, H, W) - 多普勒区域掩码
        """
        # 1. 垂直方向能量集中（速度轴）
        # 对列求平均，得到垂直方向的能量分布
        vertical_energy = log_magnitude.mean(dim=-1, keepdim=True)  # (B, 1, H, 1)
        vertical_threshold = torch.quantile(vertical_energy.flatten(1), 0.8, dim=1, keepdim=True)  # (B, 1)
        vertical_threshold = vertical_threshold.view(-1, 1, 1, 1)
        vertical_mask = (vertical_energy > vertical_threshold).float()
        
        # 2. 水平方向能量集中（距离轴）
        # 对行求平均，得到水平方向的能量分布
        horizontal_energy = log_magnitude.mean(dim=-2, keepdim=True)  # (B, 1, 1, W)
        horizontal_threshold = torch.quantile(horizontal_energy.flatten(1), 0.8, dim=1, keepdim=True)  # (B, 1)
        horizontal_threshold = horizontal_threshold.view(-1, 1, 1, 1)
        horizontal_mask = (horizontal_energy > horizontal_threshold).float()
        
        # 3. 多普勒十字 = 垂直 + 水平
        doppler_mask = vertical_mask + horizontal_mask
        doppler_mask = torch.clamp(doppler_mask, 0, 1)
        
        return doppler_mask
    
    def forward(self, image):
        """
        提取多普勒特征
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            features: (B, feature_dim) - 多普勒特征
        """
        # 1. FFT变换到频域
        fft = torch.fft.rfft2(image, norm='ortho')
        magnitude = torch.abs(fft)
        
        # 2. 对数幅度谱（突出多普勒特征）
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 3. 检测多普勒十字区域
        doppler_mask = self.detect_doppler_cross(log_magnitude)
        
        # 4. 提取多普勒区域
        doppler_region = doppler_mask * log_magnitude
        
        # 5. 特征提取
        features = self.feature_net(doppler_region)
        
        return features


class ClutterFeatureExtractor(nn.Module):
    """
    地杂波特征提取器
    提取中间那条线（地杂波）- 次要
    """
    def __init__(self, base_channels=32):
        super().__init__()
        
        # 空间特征提取（中间那条线）
        self.spatial_net = nn.Sequential(
            nn.Conv1d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels)
        )
        
        # 频域特征提取（低频区域）
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
    
    def create_low_frequency_mask(self, shape, device):
        """
        创建低频频域掩码（地杂波主要在低频）
        
        Args:
            shape: (B, 1, H, W) - 频域形状
            device: 设备
        
        Returns:
            low_freq_mask: (1, 1, H, W) - 低频掩码
        """
        B, C, H, W = shape
        
        # 中心位置
        center_h = H // 2
        center_w = W // 2
        
        # 创建坐标网格
        y_coords = torch.arange(H, device=device).float().view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device).float().view(1, 1, 1, W)
        
        # 距离中心的距离
        y_dist = torch.abs(y_coords - center_h)
        x_dist = torch.abs(x_coords - center_w)
        
        # 低频区域（中心20%区域）
        low_freq_ratio = 0.2
        threshold_h = H * low_freq_ratio
        threshold_w = W * low_freq_ratio
        
        low_freq_mask = ((y_dist < threshold_h) & (x_dist < threshold_w)).float()
        
        return low_freq_mask
    
    def extract_center_line(self, image):
        """
        提取中间那条线（地杂波）
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            line_features: (B, feature_dim) - 地杂波特征
        """
        B, C, H, W = image.shape
        center_row = H // 2
        
        # 提取中间几行（中心±2行，共5行）
        # 确保索引不越界
        start_row = max(0, center_row - 2)
        end_row = min(H, center_row + 3)
        line_region = image[:, :, start_row:end_row, :]  # (B, 1, n, W)
        
        # 对行求平均，得到一条线
        line = line_region.mean(dim=2)  # (B, 1, W)
        
        # 1D卷积提取特征
        line_features = self.spatial_net(line)  # (B, feature_dim)
        
        return line_features
    
    def extract_low_frequency(self, image):
        """
        提取低频频域特征（地杂波）
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            freq_features: (B, feature_dim) - 低频特征
        """
        # FFT变换
        fft = torch.fft.rfft2(image, norm='ortho')
        magnitude = torch.abs(fft)
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # 创建低频掩码
        low_freq_mask = self.create_low_frequency_mask(log_magnitude.shape, log_magnitude.device)
        
        # 提取低频区域
        low_freq_region = low_freq_mask * log_magnitude
        
        # 特征提取
        freq_features = self.frequency_net(low_freq_region)
        
        return freq_features
    
    def forward(self, image):
        """
        提取地杂波特征
        
        Args:
            image: (B, 1, H, W) - 输入图像
        
        Returns:
            features: (B, feature_dim) - 地杂波特征
        """
        # 1. 空间特征（中间那条线）
        spatial_features = self.extract_center_line(image)
        
        # 2. 频域特征（低频区域）
        freq_features = self.extract_low_frequency(image)
        
        # 3. 特征融合
        combined = torch.cat([spatial_features, freq_features], dim=1)
        features = self.fusion(combined)
        
        return features


class FeatureDiscriminator(nn.Module):
    """
    特征判别器
    计算真实特征和生成特征的差别
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        
        # 差别计算网络
        self.difference_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, real_features, fake_features):
        """
        计算特征差别
        
        Args:
            real_features: (B, feature_dim) - 真实特征
            fake_features: (B, feature_dim) - 生成特征
        
        Returns:
            diff: (B, 1) - 差别分数（越小越好）
        """
        # 方法1: 直接L2距离
        diff_l2 = torch.norm(real_features - fake_features, p=2, dim=1, keepdim=True)
        
        # 方法2: 通过网络学习差别（可选）
        # combined = torch.cat([real_features, fake_features], dim=1)
        # diff_net = self.difference_net(combined)
        
        # 使用L2距离（更稳定）
        return diff_l2.mean()


class DopplerClutterDiscriminator(nn.Module):
    """
    多普勒+地杂波判别器
    重点：多普勒效应（主要，权重0.75）
    次要：地杂波（中间那条线，权重0.25）
    """
    def __init__(self, doppler_weight=0.75, clutter_weight=0.25):
        super().__init__()
        
        # 权重
        self.doppler_weight = doppler_weight
        self.clutter_weight = clutter_weight
        
        # 1. 多普勒特征提取器（主要）
        self.doppler_extractor = DopplerFeatureExtractor(base_channels=64)
        
        # 2. 地杂波特征提取器（次要）
        self.clutter_extractor = ClutterFeatureExtractor(base_channels=32)
        
        # 3. 多普勒特征判别器
        doppler_feature_dim = 64 * 2  # base_channels * 2
        self.doppler_discriminator = FeatureDiscriminator(feature_dim=doppler_feature_dim)
        
        # 4. 地杂波特征判别器
        clutter_feature_dim = 32  # base_channels
        self.clutter_discriminator = FeatureDiscriminator(feature_dim=clutter_feature_dim)
    
    def forward(self, real_image, fake_image):
        """
        计算真实图和生成图的差别
        
        Args:
            real_image: (B, 1, H, W) - 真实RD图
            fake_image: (B, 1, H, W) - 生成RD图
        
        Returns:
            result: dict - 包含总差别、多普勒差别、地杂波差别
        """
        # 1. 提取多普勒特征（主要）
        real_doppler = self.doppler_extractor(real_image)
        fake_doppler = self.doppler_extractor(fake_image)
        
        # 2. 提取地杂波特征（次要）
        real_clutter = self.clutter_extractor(real_image)
        fake_clutter = self.clutter_extractor(fake_image)
        
        # 3. 计算多普勒差别（主要）
        doppler_diff = self.doppler_discriminator(real_doppler, fake_doppler)
        
        # 4. 计算地杂波差别（次要）
        clutter_diff = self.clutter_discriminator(real_clutter, fake_clutter)
        
        # 5. 综合差别（多普勒权重高，地杂波权重低）
        total_diff = (
            self.doppler_weight * doppler_diff +
            self.clutter_weight * clutter_diff
        )
        
        return {
            'total_diff': total_diff,
            'doppler_diff': doppler_diff,
            'clutter_diff': clutter_diff
        }


def doppler_clutter_gan_loss(discriminator, real_image, fake_image):
    """
    多普勒+地杂波GAN损失
    
    Args:
        discriminator: DopplerClutterDiscriminator
        real_image: (B, 1, H, W) - 真实图像
        fake_image: (B, 1, H, W) - 生成图像
    
    Returns:
        loss: scalar - GAN损失
        diffs: dict - 差别详情
    """
    diffs = discriminator(real_image, fake_image)
    
    # GAN损失（生成器希望最小化差别）
    loss = diffs['total_diff']
    
    return loss, diffs


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("DopplerClutterDiscriminator 测试")
    print("="*60)
    
    # 创建判别器
    discriminator = DopplerClutterDiscriminator(
        doppler_weight=0.75,
        clutter_weight=0.25
    )
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512)
    fake_image = torch.randn(batch_size, 1, 512, 512)
    
    # 前向传播
    diffs = discriminator(real_image, fake_image)
    
    print(f"\n输入形状: real={real_image.shape}, fake={fake_image.shape}")
    print(f"\n差别结果:")
    print(f"  总差别: {diffs['total_diff'].item():.6f}")
    print(f"  多普勒差别: {diffs['doppler_diff'].item():.6f}")
    print(f"  地杂波差别: {diffs['clutter_diff'].item():.6f}")
    
    # 测试损失
    loss, diffs_detail = doppler_clutter_gan_loss(discriminator, real_image, fake_image)
    print(f"\nGAN损失: {loss.item():.6f}")
    
    print("="*60)

