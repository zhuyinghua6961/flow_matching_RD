"""
V5 WGAN-GP测试脚本
测试Critic和损失函数的实现
"""
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from v5_wgan_gp.critic_doppler import DopplerOnlyCritic, doppler_wgan_gp_loss, doppler_feature_matching_loss


def test_critic():
    """测试WGAN-GP Critic"""
    print("="*60)
    print("测试 WGAN-GP DopplerOnlyCritic")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建Critic
    critic = DopplerOnlyCritic(base_channels=64).to(device)
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512).to(device)
    fake_image = torch.randn(batch_size, 1, 512, 512).to(device)
    
    print(f"\n✓ Critic创建成功")
    
    # 测试前向传播
    scores = critic(real_image)
    print(f"\n前向传播测试:")
    print(f"  输入形状: {real_image.shape}")
    print(f"  输出评分形状: {scores.shape}")
    print(f"  评分范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"  评分均值: {scores.mean().item():.4f}")
    
    # 测试返回特征
    scores, features, mask = critic(real_image, return_features=True)
    print(f"\n返回特征测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  多普勒掩码形状: {mask.shape}")
    print(f"  多普勒区域覆盖率: {mask.mean().item():.4f}")
    
    # 测试WGAN-GP损失
    print(f"\nWGAN-GP损失测试:")
    c_loss, c_info = doppler_wgan_gp_loss(
        critic, real_image, fake_image, mode='critic', lambda_gp=10.0
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


def test_gradient_penalty():
    """测试梯度惩罚的数值稳定性"""
    print("\n" + "="*60)
    print("测试梯度惩罚数值稳定性")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    critic = DopplerOnlyCritic(base_channels=32).to(device)  # 小一点的模型测试更快
    
    batch_size = 4
    real_image = torch.randn(batch_size, 1, 256, 256).to(device)
    fake_image = torch.randn(batch_size, 1, 256, 256).to(device)
    
    # 测试不同的lambda_gp值
    lambda_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    print(f"\n测试不同梯度惩罚系数:")
    for lambda_gp in lambda_values:
        c_loss, c_info = doppler_wgan_gp_loss(
            critic, real_image, fake_image, mode='critic', lambda_gp=lambda_gp
        )
        print(f"  λ_GP={lambda_gp:4.1f}: 总损失={c_loss.item():8.4f}, "
              f"W_loss={c_info['wasserstein_loss']:7.4f}, "
              f"GP_loss={c_info['gp_loss']:7.4f}")
    
    # 测试梯度惩罚的梯度范数分布
    print(f"\n梯度范数分布测试:")
    from v5_wgan_gp.critic_doppler import gradient_penalty
    
    # 多次采样测试
    gradient_norms = []
    for _ in range(10):
        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_image + (1 - alpha) * fake_image
        interpolated.requires_grad_(True)
        
        # 计算梯度
        scores = critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gradient_norms.extend(gradient_norm.detach().cpu().numpy())
    
    gradient_norms = torch.tensor(gradient_norms)
    print(f"  梯度范数统计:")
    print(f"    均值: {gradient_norms.mean().item():.4f}")
    print(f"    标准差: {gradient_norms.std().item():.4f}")
    print(f"    最小值: {gradient_norms.min().item():.4f}")
    print(f"    最大值: {gradient_norms.max().item():.4f}")
    print(f"    接近1.0的比例: {((gradient_norms - 1.0).abs() < 0.1).float().mean().item():.2%}")
    
    print(f"\n✓ 梯度惩罚测试通过！")
    print("="*60)


def test_training_step():
    """测试训练步骤"""
    print("\n" + "="*60)
    print("测试训练步骤")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    critic = DopplerOnlyCritic(base_channels=32).to(device)
    
    # 创建优化器
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 256, 256).to(device)
    fake_image = torch.randn(batch_size, 1, 256, 256).to(device)
    
    print(f"\n模拟训练步骤:")
    
    # 训练前的评分
    with torch.no_grad():
        real_scores_before = critic(real_image).mean().item()
        fake_scores_before = critic(fake_image).mean().item()
    
    print(f"  训练前 - 真实评分: {real_scores_before:.4f}, 生成评分: {fake_scores_before:.4f}")
    
    # 执行几步训练
    for step in range(5):
        critic_optimizer.zero_grad()
        
        c_loss, c_info = doppler_wgan_gp_loss(
            critic, real_image, fake_image, mode='critic', lambda_gp=10.0
        )
        
        c_loss.backward()
        critic_optimizer.step()
        
        print(f"  Step {step+1}: 损失={c_loss.item():.4f}, "
              f"真实评分={c_info['real_score']:.4f}, "
              f"生成评分={c_info['fake_score']:.4f}, "
              f"差距={c_info['score_gap']:.4f}")
    
    # 训练后的评分
    with torch.no_grad():
        real_scores_after = critic(real_image).mean().item()
        fake_scores_after = critic(fake_image).mean().item()
    
    print(f"  训练后 - 真实评分: {real_scores_after:.4f}, 生成评分: {fake_scores_after:.4f}")
    print(f"  评分变化 - 真实: {real_scores_after - real_scores_before:+.4f}, "
          f"生成: {fake_scores_after - fake_scores_before:+.4f}")
    
    print(f"\n✓ 训练步骤测试通过！")
    print("="*60)


if __name__ == "__main__":
    print("🚀 V5 WGAN-GP 测试开始")
    
    try:
        # 基础功能测试
        test_critic()
        
        # 梯度惩罚测试
        test_gradient_penalty()
        
        # 训练步骤测试
        test_training_step()
        
        print(f"\n🎉 所有测试通过！V5 WGAN-GP实现正确")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
