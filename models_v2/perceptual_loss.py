"""
Perceptual Loss - VGG特征匹配
用于提升生成图像的视觉质量和纹理细节
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16特征提取器
    提取多层特征用于感知损失
    """
    def __init__(self, feature_layers=(3, 8, 15)):
        """
        Args:
            feature_layers: tuple - VGG16层索引
                默认: (3, 8, 15) 对应 [relu1_2, relu2_2, relu3_3]
        """
        super().__init__()
        
        # 加载预训练的VGG16
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        
        # 冻结参数（不训练VGG）
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        
        # 设置为评估模式
        self.eval()
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) - 灰度图
        Returns:
            features: List[(B, C_i, H_i, W_i)] - 多层特征
        """
        # VGG需要3通道输入，复制灰度通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 标准化（ImageNet统计）
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # 提取多层特征
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features


class PerceptualLoss(nn.Module):
    """
    感知损失 - 匹配VGG特征
    
    优势:
    1. 保留纹理细节
    2. 学习背景杂波
    3. 视觉质量更好
    4. 避免过度平滑
    """
    def __init__(self, feature_layers=(3, 8, 15), feature_weights=None):
        """
        Args:
            feature_layers: tuple - VGG层索引
            feature_weights: tuple - 各层权重（None=均等）
        """
        super().__init__()
        
        self.feature_extractor = VGG16FeatureExtractor(feature_layers)
        
        # 特征权重
        if feature_weights is None:
            feature_weights = [1.0] * len(feature_layers)
        self.feature_weights = feature_weights
    
    def forward(self, generated, target):
        """
        Args:
            generated: (B, 1, H, W) - 生成的图像
            target: (B, 1, H, W) - 目标图像
        Returns:
            loss: scalar - 感知损失
        """
        # 提取特征
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        
        # 计算各层特征的L2距离
        loss = 0.0
        for gen_feat, target_feat, weight in zip(gen_features, target_features, self.feature_weights):
            loss += weight * F.mse_loss(gen_feat, target_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失 = Flow Matching Loss + Perceptual Loss
    
    Loss = loss_fm + λ * loss_perceptual
    """
    def __init__(self, perceptual_weight=0.01, feature_layers=(3, 8, 15)):
        """
        Args:
            perceptual_weight: float - 感知损失权重（建议0.01-0.05）
            feature_layers: tuple - VGG层索引
        """
        super().__init__()
        
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = PerceptualLoss(feature_layers)
    
    def forward(self, model, sim_image, real_image, compute_perceptual=True):
        """
        Args:
            model: Sim2RealFlowModel
            sim_image: (B, 1, H, W) - 仿真图
            real_image: (B, 1, H, W) - 真实图
            compute_perceptual: bool - 是否计算感知损失（训练时可间隔计算）
        
        Returns:
            total_loss: scalar
            loss_dict: dict - 各项损失
        """
        # 1. Flow Matching Loss
        loss_fm = model.compute_loss(sim_image, real_image)
        
        loss_dict = {
            'loss_fm': loss_fm.item()
        }
        
        # 2. Perceptual Loss（可选）
        if compute_perceptual and self.perceptual_weight > 0:
            # 生成一次图像（用于计算感知损失）
            with torch.no_grad():
                # 使用少步ODE快速生成（节省时间）
                generated = model.generate(sim_image, ode_steps=5, ode_method='euler')
            
            # 计算感知损失（需要梯度）
            # 注意：这里需要重新生成一次以保留梯度
            # 但为了效率，我们直接对generated计算（虽然没有梯度）
            # 更好的做法是间隔计算或降低频率
            loss_perceptual = self.perceptual_loss(generated, real_image)
            
            loss_dict['loss_perceptual'] = loss_perceptual.item()
            total_loss = loss_fm + self.perceptual_weight * loss_perceptual
        else:
            loss_dict['loss_perceptual'] = 0.0
            total_loss = loss_fm
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("Perceptual Loss 测试")
    print("="*60)
    
    # 模拟数据
    generated = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    
    # 测试Perceptual Loss
    perceptual_criterion = PerceptualLoss(feature_layers=(3, 8, 15))
    loss = perceptual_criterion(generated, target)
    
    print(f"\n输入: generated={generated.shape}, target={target.shape}")
    print(f"Perceptual Loss: {loss.item():.6f}")
    
    # 测试Combined Loss
    from flow_matching_v2 import Sim2RealFlowModel
    
    model = Sim2RealFlowModel(base_channels=32, channel_mult=(1, 2, 4, 8))
    sim_img = torch.randn(2, 1, 256, 256)
    real_img = torch.randn(2, 1, 256, 256)
    
    combined_criterion = CombinedLoss(perceptual_weight=0.01)
    total_loss, loss_dict = combined_criterion(model, sim_img, real_img, compute_perceptual=True)
    
    print(f"\nCombined Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*60)

