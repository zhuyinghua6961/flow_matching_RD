"""
Flow Matching完整模型
整合UNet + ControlNet + ODE求解
"""
import torch
import torch.nn as nn
from .unet import UNet
from .controlnet import ControlNet


class FlowMatchingModel(nn.Module):
    """
    完整的Flow Matching模型
    
    训练时：预测速度场 v(x_t, t, cond)
    推理时：通过ODE求解生成真实RD图
    """
    def __init__(
        self,
        # UNet参数
        unet_base_channels=64,
        unet_channel_mult=(1, 2, 4, 8),
        # ControlNet参数
        controlnet_base_channels=32,
        controlnet_channel_mult=(1, 2, 4, 8),
        # 共享参数
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.0,
        time_emb_dim=256
    ):
        super().__init__()
        
        # UNet主干
        self.unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            dropout=dropout,
            time_emb_dim=time_emb_dim
        )
        
        # ControlNet分支
        self.controlnet = ControlNet(
            cond_channels=4,  # sim_RD(3) + heatmap(1)
            base_channels=controlnet_base_channels,
            channel_mult=controlnet_channel_mult,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            dropout=dropout,
            time_emb_dim=time_emb_dim
        )

    def forward(self, x_t, time, sim_rd, heatmap):
        """
        前向传播（训练时使用）
        
        Args:
            x_t: (B, 3, H, W) 插值后的噪声图像
            time: (B,) 时间步 [0, 1]
            sim_rd: (B, 3, H, W) 仿真RD图
            heatmap: (B, 1, H, W) 热力图
        
        Returns:
            v_pred: (B, 3, H, W) 预测的速度场
        """
        # 构建ControlNet条件
        cond = torch.cat([sim_rd, heatmap], dim=1)  # (B, 4, H, W)
        
        # ControlNet前向
        controlnet_outputs = self.controlnet(cond, time)
        
        # UNet前向（注入ControlNet输出）
        v_pred = self.unet(x_t, time, controlnet_outputs)
        
        return v_pred

    @torch.no_grad()
    def sample(self, sim_rd, heatmap, num_steps=20, method='euler'):
        """
        通过ODE求解生成真实RD图（推理时使用）
        
        Args:
            sim_rd: (B, 3, H, W) 仿真RD图
            heatmap: (B, 1, H, W) 热力图
            num_steps: ODE求解步数
            method: 'euler' 或 'rk4'
        
        Returns:
            real_rd: (B, 3, H, W) 生成的真实RD图
        """
        device = sim_rd.device
        batch_size = sim_rd.shape[0]
        
        # 初始化：x_0 = sim_rd
        x = sim_rd.clone()
        
        # 时间步长
        dt = 1.0 / num_steps
        
        # ODE求解
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            if method == 'euler':
                # 一阶欧拉法
                v = self.forward(x, t, sim_rd, heatmap)
                x = x + v * dt
            
            elif method == 'rk4':
                # 四阶龙格库塔法（更精确但更慢）
                k1 = self.forward(x, t, sim_rd, heatmap)
                k2 = self.forward(x + 0.5 * dt * k1, t + 0.5 * dt, sim_rd, heatmap)
                k3 = self.forward(x + 0.5 * dt * k2, t + 0.5 * dt, sim_rd, heatmap)
                k4 = self.forward(x + dt * k3, t + dt, sim_rd, heatmap)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x

    def get_num_parameters(self):
        """返回模型参数量统计"""
        unet_params = sum(p.numel() for p in self.unet.parameters())
        controlnet_params = sum(p.numel() for p in self.controlnet.parameters())
        total_params = unet_params + controlnet_params
        
        return {
            'unet': unet_params / 1e6,
            'controlnet': controlnet_params / 1e6,
            'total': total_params / 1e6
        }


if __name__ == "__main__":
    # 测试
    model = FlowMatchingModel(
        unet_base_channels=64,
        unet_channel_mult=(1, 2, 4, 8),
        controlnet_base_channels=32,
        controlnet_channel_mult=(1, 2, 4, 8)
    )
    
    # 训练模式测试
    batch_size = 2
    x_t = torch.randn(batch_size, 3, 512, 512)
    time = torch.rand(batch_size)
    sim_rd = torch.randn(batch_size, 3, 512, 512)
    heatmap = torch.randn(batch_size, 1, 512, 512)
    
    v_pred = model(x_t, time, sim_rd, heatmap)
    print(f"训练模式 - 输入: {x_t.shape}, 输出: {v_pred.shape}")
    
    # 推理模式测试
    real_rd = model.sample(sim_rd, heatmap, num_steps=10)
    print(f"推理模式 - 输入: {sim_rd.shape}, 输出: {real_rd.shape}")
    
    # 参数量统计
    params = model.get_num_parameters()
    print(f"\n模型参数量:")
    print(f"  UNet: {params['unet']:.2f}M")
    print(f"  ControlNet: {params['controlnet']:.2f}M")
    print(f"  Total: {params['total']:.2f}M")

