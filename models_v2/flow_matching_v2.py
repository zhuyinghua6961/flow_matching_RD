"""
Flow Matching V2 - 纯图像对的Sim2Real模型
无需prompt，端到端学习
"""
import torch
import torch.nn as nn
from .sim_encoder import SimEncoder
from .conditional_unet import ConditionalUNet


class Sim2RealFlowModel(nn.Module):
    """
    Conditional Flow Matching for Sim2Real
    
    架构:
        sim_image → SimEncoder → cond_features
                                      ↓
        (x_t, t, cond_features) → ConditionalUNet → velocity_field
    
    训练:
        1. 随机采样时间 t ~ Uniform(0, 1)
        2. 构造 x_t = (1-t)*noise + t*real_image
        3. 预测速度场 v = model(x_t, t, sim_image)
        4. 真实速度场 v_true = real_image - noise
        5. Loss = MSE(v, v_true)
    
    推理:
        1. 初始化 x_0 ~ N(0, I)
        2. ODE求解: dx/dt = v(x, t, sim_image)
        3. 从 t=0 到 t=1, 得到 x_1 ≈ real_image
    """
    def __init__(
        self,
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),
        time_embed_dim=256,
        num_res_blocks=2,
        attention_levels=(3,),
        dropout=0.0
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        
        # SimEncoder: 提取仿真图特征
        self.sim_encoder = SimEncoder(
            in_channels=1,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        # ConditionalUNet: 条件去噪网络
        self.denoiser = ConditionalUNet(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            time_embed_dim=time_embed_dim,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            dropout=dropout
        )
    
    def forward(self, x_t, t, sim_image):
        """
        前向传播 - 预测速度场
        
        Args:
            x_t: (B, 1, 512, 512) - 时间t的噪声图
            t: (B,) - 时间步，范围[0, 1]
            sim_image: (B, 1, 512, 512) - 仿真图（条件）
        
        Returns:
            v: (B, 1, 512, 512) - 预测的速度场
        """
        # 提取仿真图的多尺度特征
        cond_features = self.sim_encoder(sim_image)
        
        # 基于条件特征预测速度场
        v = self.denoiser(x_t, t, cond_features)
        
        return v
    
    @torch.no_grad()
    def generate(self, sim_image, ode_steps=50, ode_method='euler'):
        """
        推理 - 从仿真图生成真实图
        
        Args:
            sim_image: (B, 1, 512, 512) - 仿真图
            ode_steps: int - ODE求解步数
            ode_method: str - 'euler' 或 'rk4'
        
        Returns:
            real_image: (B, 1, 512, 512) - 生成的真实图
        """
        device = sim_image.device
        batch_size = sim_image.shape[0]
        
        # 初始化：纯噪声
        x = torch.randn_like(sim_image)
        
        # ODE求解：从 t=0 到 t=1
        dt = 1.0 / ode_steps
        
        if ode_method == 'euler':
            # Euler方法
            for i in range(ode_steps):
                t = torch.ones(batch_size, device=device) * (i * dt)
                v = self.forward(x, t, sim_image)
                x = x + v * dt
        
        elif ode_method == 'rk4':
            # Runge-Kutta 4阶方法（更精确）
            for i in range(ode_steps):
                t = torch.ones(batch_size, device=device) * (i * dt)
                
                k1 = self.forward(x, t, sim_image)
                k2 = self.forward(x + 0.5 * dt * k1, t + 0.5 * dt, sim_image)
                k3 = self.forward(x + 0.5 * dt * k2, t + 0.5 * dt, sim_image)
                k4 = self.forward(x + dt * k3, t + dt, sim_image)
                
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        else:
            raise ValueError(f"Unknown ODE method: {ode_method}")
        
        return x
    
    def compute_loss(self, sim_image, real_image):
        """
        计算Flow Matching损失
        
        Args:
            sim_image: (B, 1, H, W) - 仿真图
            real_image: (B, 1, H, W) - 真实图（目标）
        
        Returns:
            loss: scalar - MSE损失
        """
        batch_size = sim_image.shape[0]
        device = sim_image.device
        
        # 1. 随机采样时间 t ∈ (ε, 1)
        t = torch.rand(batch_size, device=device) * (1 - 1e-5) + 1e-5
        
        # 2. 构造 x_t (线性插值路径)
        noise = torch.randn_like(real_image)  # x_0 ~ N(0, I)
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * noise + t_expand * real_image
        
        # 3. 预测速度场
        v_pred = self.forward(x_t, t, sim_image)
        
        # 4. 真实速度场（从noise到real_image的方向）
        v_true = real_image - noise
        
        # 5. MSE Loss
        loss = torch.nn.functional.mse_loss(v_pred, v_true)
        
        return loss


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("Sim2RealFlowModel V2 测试")
    print("="*60)
    
    model = Sim2RealFlowModel(
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),
        attention_levels=(3,)
    )
    
    # 模拟数据
    batch_size = 2
    sim_img = torch.randn(batch_size, 1, 512, 512)
    real_img = torch.randn(batch_size, 1, 512, 512)
    
    # 测试训练
    print("\n[训练模式]")
    loss = model.compute_loss(sim_img, real_img)
    print(f"  Flow Matching Loss: {loss.item():.6f}")
    
    # 测试推理
    print("\n[推理模式]")
    model.eval()
    generated = model.generate(sim_img, ode_steps=10, ode_method='euler')
    print(f"  输入: sim_img={sim_img.shape}")
    print(f"  输出: generated={generated.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.sim_encoder.parameters())
    denoiser_params = sum(p.numel() for p in model.denoiser.parameters())
    
    print(f"\n[参数统计]")
    print(f"  SimEncoder: {encoder_params:,}")
    print(f"  ConditionalUNet: {denoiser_params:,}")
    print(f"  总计: {total_params:,}")
    
    print("\n" + "="*60)

