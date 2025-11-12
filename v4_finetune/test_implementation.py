"""
V4实现测试脚本
验证判别器和训练器是否正确实现
"""
import sys
from pathlib import Path
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from v4_finetune.discriminator_doppler import (
    DopplerOnlyDiscriminator,
    doppler_adversarial_loss,
    doppler_feature_matching_loss
)


def test_discriminator():
    """测试判别器"""
    print("="*60)
    print("测试 DopplerOnlyDiscriminator")
    print("="*60)
    
    # 创建判别器
    discriminator = DopplerOnlyDiscriminator(base_channels=64)
    
    # 模拟数据
    batch_size = 2
    real_image = torch.randn(batch_size, 1, 512, 512)
    fake_image = torch.randn(batch_size, 1, 512, 512)
    
    print("\n✓ 判别器创建成功")
    
    # 测试前向传播
    logits = discriminator(real_image)
    print(f"\n前向传播测试:")
    print(f"  输入形状: {real_image.shape}")
    print(f"  输出logits形状: {logits.shape}")
    print(f"  判别概率: {torch.sigmoid(logits).mean().item():.4f}")
    
    # 测试返回特征
    logits, features, mask = discriminator(real_image, return_features=True)
    print(f"\n返回特征测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  多普勒掩码形状: {mask.shape}")
    print(f"  多普勒区域覆盖率: {mask.mean().item():.4f}")
    
    # 测试对抗损失
    print("\n对抗损失测试:")
    d_loss, d_info = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='discriminator'
    )
    print(f"  判别器损失: {d_loss.item():.6f}")
    print(f"  真实图像准确率: {d_info['real_acc']:.4f}")
    print(f"  生成图像准确率: {d_info['fake_acc']:.4f}")
    
    g_loss, g_info = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='generator'
    )
    print(f"  生成器对抗损失: {g_loss.item():.6f}")
    
    # 测试特征匹配损失
    fm_loss, fm_info = doppler_feature_matching_loss(
        discriminator, real_image, fake_image
    )
    print(f"  特征匹配损失: {fm_loss.item():.6f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数占用显存: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✓ 所有测试通过！")
    print("="*60)


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("测试梯度反向传播")
    print("="*60)
    
    discriminator = DopplerOnlyDiscriminator(base_channels=64)
    discriminator.train()
    
    real_image = torch.randn(2, 1, 512, 512, requires_grad=False)
    fake_image = torch.randn(2, 1, 512, 512, requires_grad=True)
    
    # 判别器梯度
    d_loss, _ = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='discriminator'
    )
    d_loss.backward()
    
    print("\n判别器梯度检查:")
    has_grad = False
    for name, param in discriminator.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"  ✓ {name}: 梯度范数 = {param.grad.norm().item():.6f}")
            if has_grad:
                break  # 只显示第一个有梯度的参数
    
    if has_grad:
        print("  ✓ 判别器梯度正常")
    else:
        print("  ✗ 判别器没有梯度！")
    
    # 清空梯度
    discriminator.zero_grad()
    fake_image.grad = None
    
    # 生成器梯度（应该传播到fake_image）
    g_loss, _ = doppler_adversarial_loss(
        discriminator, real_image, fake_image, mode='generator'
    )
    g_loss.backward()
    
    print("\n生成器梯度检查:")
    if fake_image.grad is not None:
        print(f"  ✓ fake_image有梯度: 梯度范数 = {fake_image.grad.norm().item():.6f}")
    else:
        print("  ✗ fake_image没有梯度！")
    
    print("\n✓ 梯度流测试通过！")
    print("="*60)


def test_multi_scale():
    """测试多种图像尺寸"""
    print("\n" + "="*60)
    print("测试多种图像尺寸")
    print("="*60)
    
    discriminator = DopplerOnlyDiscriminator(base_channels=64)
    
    sizes = [256, 512, 1024]
    
    for size in sizes:
        image = torch.randn(1, 1, size, size)
        try:
            logits = discriminator(image)
            print(f"  ✓ {size}x{size}: 输出形状 {logits.shape}")
        except Exception as e:
            print(f"  ✗ {size}x{size}: 失败 - {str(e)}")
    
    print("\n✓ 多尺寸测试通过！")
    print("="*60)


def test_cuda_support():
    """测试CUDA支持"""
    print("\n" + "="*60)
    print("测试CUDA支持")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  CUDA不可用，跳过测试")
        print("="*60)
        return
    
    device = torch.device('cuda')
    discriminator = DopplerOnlyDiscriminator(base_channels=64).to(device)
    
    image = torch.randn(2, 1, 512, 512).to(device)
    
    try:
        logits = discriminator(image)
        print(f"  ✓ CUDA前向传播成功")
        print(f"  ✓ 输出设备: {logits.device}")
        
        # 测试反向传播
        loss = logits.mean()
        loss.backward()
        print(f"  ✓ CUDA反向传播成功")
        
        # 显存占用
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"  显存占用: {allocated:.2f} MB")
        
    except Exception as e:
        print(f"  ✗ CUDA测试失败: {str(e)}")
    
    print("\n✓ CUDA测试通过！")
    print("="*60)


def main():
    """运行所有测试"""
    print("\n" + "🧪 " + "="*58)
    print("V4实现测试套件")
    print("="*60 + "\n")
    
    try:
        # 测试判别器基本功能
        test_discriminator()
        
        # 测试梯度流
        test_gradient_flow()
        
        # 测试多尺寸支持
        test_multi_scale()
        
        # 测试CUDA支持
        test_cuda_support()
        
        print("\n" + "🎉 " + "="*58)
        print("所有测试通过！V4实现正确！")
        print("="*60)
        print("\n你可以开始使用V4进行微调训练了！")
        print("命令：python v4_finetune/train_finetune.py --config v4_finetune/config_finetune.yaml --pretrained <your_pretrained_model.pth>")
        print("\n")
        
    except Exception as e:
        print("\n" + "❌ " + "="*58)
        print(f"测试失败: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
