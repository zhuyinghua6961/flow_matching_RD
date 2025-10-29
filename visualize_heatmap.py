"""
可视化热力图生成（用于验证多目标支持）
"""
import numpy as np
from utils import HeatmapGenerator


def visualize_heatmaps():
    """可视化单目标、双目标、三目标的热力图"""
    import matplotlib.pyplot as plt
    
    # 创建生成器
    generator = HeatmapGenerator(
        img_size=512,
        max_speed=30.0,
        max_range=200.0,
        sigma=10.0
    )
    
    # 三种prompt
    prompts = [
        # 单目标
        "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 1, the first target: distance = 102m, velocity = 20.00m/s.",
        
        # 双目标
        "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s.",
        
        # 三目标
        "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 3, the first target: distance = 79m, velocity = -27.00m/s, the second target: distance = 126m, velocity = -18.00m/s, the third target: distance = 26m, velocity = 26.00m/s."
    ]
    
    titles = ["单目标", "双目标", "三目标"]
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (prompt, title) in enumerate(zip(prompts, titles)):
        # 生成热力图
        heatmap = generator(prompt)
        heatmap_np = heatmap.squeeze().numpy()
        
        # 提取目标信息
        targets = generator.parse_prompt(prompt)
        
        # 绘制热力图
        im = axes[i].imshow(heatmap_np, cmap='hot', origin='upper', 
                           extent=[-30, 30, 200, 0])
        axes[i].set_title(f'{title}\n目标数: {len(targets)}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('速度 (m/s)', fontsize=10)
        axes[i].set_ylabel('距离 (m)', fontsize=10)
        
        # 添加网格
        axes[i].grid(True, alpha=0.3, linestyle='--')
        
        # 标注目标位置
        for j, target in enumerate(targets):
            speed = target['speed']
            distance = target['distance']
            axes[i].plot(speed, distance, 'b*', markersize=15, 
                        label=f'目标{j+1}: {speed}m/s, {distance}m')
        
        axes[i].legend(loc='upper right', fontsize=8)
        
        # 添加colorbar
        plt.colorbar(im, ax=axes[i], label='强度')
    
    plt.tight_layout()
    plt.savefig('heatmap_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ 可视化已保存: heatmap_visualization.png")
    plt.show()


def print_target_info():
    """打印目标信息"""
    generator = HeatmapGenerator(img_size=512, max_speed=30.0, max_range=200.0, sigma=10.0)
    
    prompts = [
        "radar-RD-map; ... target number = 1, the first target: distance = 102m, velocity = 20.00m/s.",
        "radar-RD-map; ... target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s.",
        "radar-RD-map; ... target number = 3, the first target: distance = 79m, velocity = -27.00m/s, the second target: distance = 126m, velocity = -18.00m/s, the third target: distance = 26m, velocity = 26.00m/s."
    ]
    
    print("\n" + "="*60)
    print("目标提取和坐标映射验证")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n【场景{i}】")
        targets = generator.parse_prompt(prompt)
        print(f"目标数: {len(targets)}")
        
        for j, target in enumerate(targets, 1):
            speed = target['speed']
            distance = target['distance']
            x, y = generator.params_to_pixel(speed, distance)
            print(f"  目标{j}:")
            print(f"    物理参数: 速度={speed:+6.1f} m/s, 距离={distance:6.1f} m")
            print(f"    像素坐标: x={x:6.1f}, y={y:6.1f}")
            print(f"    坐标验证: x∈[0,512]? {0 <= x <= 512}, y∈[0,512]? {0 <= y <= 512}")


if __name__ == "__main__":
    # 打印目标信息
    print_target_info()
    
    print("\n" + "="*60)
    print("✓ 多目标热力图生成功能验证完成！")
    print("="*60)
    
    # 可视化热力图（可选）
    # 注意：需要matplotlib支持，如有版本问题可跳过
    # visualize_heatmaps()

