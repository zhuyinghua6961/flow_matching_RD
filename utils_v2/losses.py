"""
Loss Functions V2 - Flow Matching损失函数
"""
import torch
import torch.nn.functional as F


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

