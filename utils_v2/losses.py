"""
Loss Functions V2 - Flow Matching损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def flow_matching_loss(model, sim_image, real_image):
    """
    Flow Matching损失
    
    原理:
        1. 构造从噪声到真实图的线性路径: x_t = (1-t)*x_0 + t*x_1
           其中 x_0 ~ N(0, I), x_1 = real_image
        
        2. 沿路径的速度场: dx/dt = x_1 - x_0 = real_image - noise
        
        3. 训练模型预测这个速度场: v_pred = model(x_t, t, sim_image)
        
        4. 损失: MSE(v_pred, v_true)
    
    Args:
        model: Sim2RealFlowModel
        sim_image: (B, 1, H, W) - 仿真图
        real_image: (B, 1, H, W) - 真实图
    
    Returns:
        loss: scalar tensor
    """
    batch_size = sim_image.shape[0]
    device = sim_image.device
    
    # 1. 随机采样时间 t ∈ (ε, 1)
    # 避免t=0和t=1的边界情况
    t = torch.rand(batch_size, device=device) * (1 - 1e-5) + 1e-5
    
    # 2. 采样初始噪声
    noise = torch.randn_like(real_image)  # x_0 ~ N(0, I)
    
    # 3. 构造 x_t (线性插值)
    t_expand = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expand) * noise + t_expand * real_image
    
    # 4. 预测速度场
    v_pred = model(x_t, t, sim_image)
    
    # 5. 真实速度场
    v_true = real_image - noise
    
    # 6. MSE Loss
    loss = F.mse_loss(v_pred, v_true)
    
    return loss


def compute_gradient_penalty(model, sim_image, real_image, lambda_gp=10.0):
    """
    梯度惩罚（可选）- 提高训练稳定性
    
    原理: 约束模型的Lipschitz常数，防止梯度爆炸
    
    Args:
        model: Sim2RealFlowModel
        sim_image: (B, 1, H, W)
        real_image: (B, 1, H, W)
        lambda_gp: float - 梯度惩罚系数
    
    Returns:
        gp_loss: scalar
    """
    batch_size = sim_image.shape[0]
    device = sim_image.device
    
    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    noise = torch.randn_like(real_image)
    interpolates = alpha * real_image + (1 - alpha) * noise
    interpolates.requires_grad_(True)
    
    # 随机时间
    t = torch.rand(batch_size, device=device)
    
    # 前向传播
    v = model(interpolates, t, sim_image)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=v,
        inputs=interpolates,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 梯度的L2范数
    gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    
    # 梯度惩罚: (||grad|| - 1)^2
    gp_loss = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    
    return gp_loss


def frequency_domain_loss(pred, target):
    """
    频域Loss - 用于学习多普勒效应等频域特征
    
    原理:
        多普勒十字效应在频域表现为特定方向的能量集中
        通过约束频域的幅度谱，强制模型学习这种结构
    
    Args:
        pred: (B, C, H, W) - 预测图像
        target: (B, C, H, W) - 目标图像
    
    Returns:
        freq_loss: scalar - 频域损失
    """
    # FFT到频域 (实数输入的2D FFT)
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    
    # 幅度谱 (magnitude spectrum)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    # 对数域的MSE (更稳定，对不同尺度的频率分量更公平)
    # log(1 + x) 避免log(0)
    pred_mag_log = torch.log(pred_mag + 1e-8)
    target_mag_log = torch.log(target_mag + 1e-8)
    
    # 幅度谱loss
    mag_loss = F.mse_loss(pred_mag_log, target_mag_log)
    
    # 可选：相位loss (但相位对训练不稳定，默认不用)
    # pred_phase = torch.angle(pred_fft)
    # target_phase = torch.angle(target_fft)
    # phase_loss = F.mse_loss(pred_phase, target_phase)
    
    return mag_loss


def ssim_loss(pred, target, window_size=11, size_average=True):
    """
    SSIM (Structural Similarity Index) Loss
    
    原理:
        关注图像的结构相似性，而非像素级差异
        对局部模式（如多普勒十字）更敏感
    
    Args:
        pred: (B, C, H, W) - 预测图像
        target: (B, C, H, W) - 目标图像
        window_size: int - 窗口大小（默认11）
        size_average: bool - 是否平均
    
    Returns:
        ssim_loss_value: scalar - SSIM损失 (1 - SSIM)
    """
    # 创建高斯窗口
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device).type_as(pred)
    
    # SSIM计算
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)


def combined_loss(
    pred, 
    target, 
    loss_weights=None,
    return_components=False
):
    """
    组合Loss - MSE + 频域 + SSIM
    
    Args:
        pred: (B, C, H, W) - 预测图像
        target: (B, C, H, W) - 目标图像
        loss_weights: dict - 各个loss的权重
            {
                'mse': 1.0,
                'frequency': 0.1,
                'ssim': 0.3
            }
        return_components: bool - 是否返回各个loss分量
    
    Returns:
        total_loss: scalar 或 dict (if return_components=True)
    """
    if loss_weights is None:
        loss_weights = {
            'mse': 1.0,
            'frequency': 0.1,
            'ssim': 0.3
        }
    
    # 1. MSE Loss (基础)
    mse = F.mse_loss(pred, target)
    
    # 2. 频域Loss (学习多普勒结构)
    freq_loss = frequency_domain_loss(pred, target)
    
    # 3. SSIM Loss (结构相似性)
    ssim = ssim_loss(pred, target)
    
    # 总loss
    total = (
        loss_weights.get('mse', 1.0) * mse +
        loss_weights.get('frequency', 0.1) * freq_loss +
        loss_weights.get('ssim', 0.3) * ssim
    )
    
    if return_components:
        return {
            'total': total,
            'mse': mse,
            'frequency': freq_loss,
            'ssim': ssim
        }
    else:
        return total


if __name__ == "__main__":
    # 测试
    from models_v2 import Sim2RealFlowModel
    
    print("="*60)
    print("Loss Functions V2 测试")
    print("="*60)
    
    model = Sim2RealFlowModel(base_channels=32, channel_mult=(1, 2, 4, 8))
    
    sim_img = torch.randn(4, 1, 256, 256)
    real_img = torch.randn(4, 1, 256, 256)
    
    # Flow Matching Loss
    loss_fm = flow_matching_loss(model, sim_img, real_img)
    print(f"\nFlow Matching Loss: {loss_fm.item():.6f}")
    
    # Gradient Penalty (可选)
    # loss_gp = compute_gradient_penalty(model, sim_img, real_img)
    # print(f"Gradient Penalty: {loss_gp.item():.6f}")
    
    print("\n" + "="*60)

