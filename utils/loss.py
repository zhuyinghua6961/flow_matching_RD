"""
优化的加权Loss - 针对RD图稀疏目标的专用损失函数
基于热力图引导的位置加权策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFlowMatchingLoss(nn.Module):
    """
    加权Flow Matching损失
    
    核心设计：
    1. 基础损失：逐像素MSE
    2. 权重图构造：
       - 灰度化热力图
       - Max Pooling下采样（保峰值）
       - 二值化阈值分割目标区域
       - 差异化赋权（目标区域30-100倍权重）
    3. 加权平均归一化（数值稳定）
    4. 可选Focal机制（关注难样本）
    
    参考图片中的Loss设计
    """
    
    def __init__(
        self,
        weight_factor=50,      # 目标区域权重因子（30-100）
        threshold=0.1,         # 热力图二值化阈值
        pool_size=8,           # Max Pooling下采样核大小（512→64）
        focal_gamma=0.0,       # Focal Loss的gamma（0表示不使用）
        use_perceptual=False,  # 是否使用感知损失
        perceptual_weight=0.1  # 感知损失权重
    ):
        super().__init__()
        
        self.weight_factor = weight_factor
        self.threshold = threshold
        self.pool_size = pool_size
        self.focal_gamma = focal_gamma
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        
        # Max Pooling（用于下采样热力图）
        self.maxpool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=pool_size
        )
        
        # 感知损失（可选）
        if use_perceptual:
            from torchvision import models
            vgg = models.vgg16(pretrained=True).features[:16]
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.eval()
    
    def construct_weight_map(self, heatmap):
        """
        构造权重图
        
        步骤：
        1. 灰度化（如果是多通道）
        2. Max Pooling下采样到64×64（保峰值）
        3. 二值化阈值
        4. 差异化赋权
        
        Args:
            heatmap: (B, 1, 512, 512) 热力图
        
        Returns:
            weight_map: (B, 1, 512, 512) 权重图
        """
        B, C, H, W = heatmap.shape
        
        # 1. 灰度化（如果需要）
        if C > 1:
            heatmap_gray = heatmap.mean(dim=1, keepdim=True)
        else:
            heatmap_gray = heatmap
        
        # 2. Max Pooling下采样（例如 512→64）
        weight_map_small = self.maxpool(heatmap_gray)  # (B, 1, 64, 64)
        
        # 3. 上采样回原尺寸（使用nearest保持峰值）
        weight_map = F.interpolate(
            weight_map_small,
            size=(H, W),
            mode='nearest'
        )
        
        # 4. 二值化：目标区域 vs 背景
        target_mask = (weight_map > self.threshold).float()
        
        # 5. 差异化赋权
        # 目标区域：weight_factor
        # 背景区域：1.0
        weight_map = target_mask * self.weight_factor + (1 - target_mask) * 1.0
        
        return weight_map, target_mask
    
    def compute_perceptual_loss(self, pred, target):
        """
        感知损失（使用VGG特征）
        
        Args:
            pred: (B, 3, H, W) 预测
            target: (B, 3, H, W) 目标
        
        Returns:
            loss: 标量
        """
        # 归一化到ImageNet范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # 提取VGG特征
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)
        
        # L2损失
        loss = F.mse_loss(pred_feat, target_feat)
        
        return loss
    
    def forward(self, v_pred, v_target, heatmap):
        """
        计算加权损失
        
        Args:
            v_pred: (B, 3, H, W) 预测的速度场
            v_target: (B, 3, H, W) 目标速度场 (real_RD - sim_RD)
            heatmap: (B, 1, H, W) 热力图
        
        Returns:
            loss_dict: 包含各项损失和指标的字典
        """
        # 1. 基础损失（逐像素MSE）
        diff = v_pred - v_target
        l_base = diff ** 2  # (B, 3, H, W)
        
        # 对通道求平均（或求和）
        l_base = l_base.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 2. 构造权重图
        weight_map, target_mask = self.construct_weight_map(heatmap)
        
        # 3. 可选Focal机制
        if self.focal_gamma > 0:
            focal = (l_base.detach() + 1e-8) ** self.focal_gamma
            weight_map = weight_map * focal
        
        # 4. 加权损失
        weighted_loss = (l_base * weight_map).sum() / (weight_map.sum() + 1e-8)
        
        # 5. 感知损失（可选）
        perceptual_loss = 0.0
        if self.use_perceptual:
            perceptual_loss = self.compute_perceptual_loss(v_pred, v_target)
        
        # 6. 总损失
        total_loss = weighted_loss + self.perceptual_weight * perceptual_loss
        
        # 7. 记录指标（用于监控）
        with torch.no_grad():
            base_loss = l_base.mean()
            
            # 目标区域损失
            target_loss = (l_base * target_mask).sum() / (target_mask.sum() + 1e-8)
            
            # 背景区域损失
            bg_mask = 1 - target_mask
            bg_loss = (l_base * bg_mask).sum() / (bg_mask.sum() + 1e-8)
            
            # 平均权重
            weight_mean = weight_map.mean()
            
            # 目标占比
            target_ratio = target_mask.mean()
        
        # 返回损失字典
        loss_dict = {
            'loss': total_loss,                    # 总损失（用于反向传播）
            'weighted_loss': weighted_loss,        # 加权损失
            'base_loss': base_loss,                # 基础MSE
            'target_loss': target_loss,            # 目标区域损失
            'bg_loss': bg_loss,                    # 背景损失
            'perceptual_loss': perceptual_loss,    # 感知损失
            'weight_mean': weight_mean,            # 平均权重
            'target_ratio': target_ratio           # 目标占比
        }
        
        return loss_dict


class SimpleMSELoss(nn.Module):
    """简单的MSE损失（用于对比实验）"""
    
    def forward(self, v_pred, v_target, heatmap=None):
        loss = F.mse_loss(v_pred, v_target)
        
        return {
            'loss': loss,
            'base_loss': loss,
            'weighted_loss': loss,
            'target_loss': torch.tensor(0.0),
            'bg_loss': torch.tensor(0.0),
            'perceptual_loss': torch.tensor(0.0),
            'weight_mean': torch.tensor(1.0),
            'target_ratio': torch.tensor(0.0)
        }


if __name__ == "__main__":
    # 测试
    print("测试加权Loss...")
    
    loss_fn = WeightedFlowMatchingLoss(
        weight_factor=50,
        threshold=0.1,
        pool_size=8,
        focal_gamma=0.0,
        use_perceptual=False
    )
    
    # 模拟数据
    batch_size = 2
    v_pred = torch.randn(batch_size, 3, 512, 512)
    v_target = torch.randn(batch_size, 3, 512, 512)
    
    # 模拟热力图（中心有高斯峰）
    heatmap = torch.zeros(batch_size, 1, 512, 512)
    center_y, center_x = 256, 256
    for i in range(batch_size):
        y, x = torch.meshgrid(
            torch.arange(512, dtype=torch.float32),
            torch.arange(512, dtype=torch.float32),
            indexing='ij'
        )
        heatmap[i, 0] = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 20**2))
    
    # 计算损失
    loss_dict = loss_fn(v_pred, v_target, heatmap)
    
    print("\n损失指标:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    print(f"\n测试通过！")

