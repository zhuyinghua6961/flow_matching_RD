"""
推理脚本 - 使用训练好的模型生成真实RD图
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from models import FlowMatchingModel
from utils import HeatmapGenerator
from config import load_config


class Inferencer:
    """推理器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # 如果检查点中没有配置，加载默认配置
            config = load_config('config.yaml')
        
        # 创建模型
        # 固定的轻量级模型架构（必须与训练时一致）
        self.model = FlowMatchingModel(
            unet_base_channels=64,
            unet_channel_mult=(1, 2, 4, 8),
            controlnet_base_channels=64,  # 与UNet保持一致
            controlnet_channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_levels=(3,),  # 只在最低分辨率(64×64)使用attention，避免OOM
            dropout=0.0,
            time_emb_dim=256
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("模型加载完成！")
        
        # 热力图生成器
        self.heatmap_generator = HeatmapGenerator(
            img_size=config.img_size,
            max_speed=config.max_speed,
            max_range=config.max_range,
            sigma=config.heatmap_sigma
        )
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.inv_transform = transforms.Normalize(
            mean=[-1, -1, -1],
            std=[2, 2, 2]
        )
        
        self.config = config
    
    def load_image(self, image_path):
        """加载并预处理图像"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # (1, 3, H, W)
        return img_tensor
    
    def auto_generate_heatmap(self, sim_img_tensor, threshold_percentile=99.5, min_distance=10):
        """
        从sim RD图自动检测目标并生成heatmap
        
        Args:
            sim_img_tensor: (1, 3, H, W) 归一化后的sim图像张量
            threshold_percentile: 亮度阈值百分位数（检测最亮的点）
            min_distance: 两个目标之间的最小距离（像素）
        
        Returns:
            heatmap: (1, 1, H, W) 生成的热力图张量
        """
        # 反归一化到[0, 1]
        img = self.inv_transform(sim_img_tensor[0]).cpu().numpy()  # (3, H, W)
        
        # 转换为灰度图（取平均）
        gray = img.mean(axis=0)  # (H, W)
        
        # 计算阈值（检测最亮的点）
        threshold = np.percentile(gray, threshold_percentile)
        
        # 二值化
        binary = gray > threshold
        
        # 查找连通区域（简单的峰值检测）
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        
        # 获取每个连通区域的质心
        centers = []
        for i in range(1, num_features + 1):
            y_coords, x_coords = np.where(labeled == i)
            if len(y_coords) > 0:
                # 使用加权质心（考虑亮度）
                weights = gray[y_coords, x_coords]
                y_center = np.average(y_coords, weights=weights)
                x_center = np.average(x_coords, weights=weights)
                centers.append((x_center, y_center))
        
        # 如果没有检测到目标，返回空热力图
        if len(centers) == 0:
            print("  ⚠️  未检测到目标，使用均匀热力图")
            heatmap = torch.ones(1, 1, self.config.img_size, self.config.img_size) * 0.1
            return heatmap.to(self.device)
        
        # 去除过近的点（非极大值抑制）
        if len(centers) > 1:
            filtered_centers = []
            centers = sorted(centers, key=lambda c: gray[int(c[1]), int(c[0])], reverse=True)
            
            for center in centers:
                too_close = False
                for existing in filtered_centers:
                    dist = np.sqrt((center[0] - existing[0])**2 + (center[1] - existing[1])**2)
                    if dist < min_distance:
                        too_close = True
                        break
                if not too_close:
                    filtered_centers.append(center)
            centers = filtered_centers
        
        print(f"  ✓ 自动检测到 {len(centers)} 个目标")
        
        # 生成多目标高斯热力图
        H, W = self.config.img_size, self.config.img_size
        heatmap = np.zeros((H, W))
        
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        sigma = self.heatmap_generator.sigma
        
        for cx, cy in centers:
            # 生成高斯峰
            gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian)  # 取最大值避免过饱和
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 转换为tensor
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        return heatmap_tensor.to(self.device)
    
    def tensor_to_image(self, tensor):
        """将tensor转换为numpy图像"""
        # 反归一化
        tensor = self.inv_transform(tensor.cpu())
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy
        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        return img
    
    @torch.no_grad()
    def inference(self, sim_rd_path, prompt=None, num_steps=20, method='euler'):
        """
        推理生成真实RD图
        
        Args:
            sim_rd_path: 仿真RD图路径
            prompt: 文本描述或参数字典（可选，如果为None则自动检测目标）
            num_steps: ODE求解步数
            method: 'euler' 或 'rk4'
        
        Returns:
            real_rd: 生成的真实RD图 (numpy array)
            heatmap: 热力图 (numpy array)
        """
        # 加载仿真图
        sim_rd = self.load_image(sim_rd_path).to(self.device)
        
        # 生成热力图
        if prompt is None:
            print("未提供prompt，自动检测目标并生成热力图...")
            heatmap = self.auto_generate_heatmap(sim_rd)
        else:
            print("使用提供的prompt生成热力图...")
            heatmap = self.heatmap_generator(prompt).unsqueeze(0).to(self.device)
        
        # 推理
        print(f"开始推理... (ODE步数: {num_steps}, 方法: {method})")
        real_rd_tensor = self.model.sample(sim_rd, heatmap, num_steps=num_steps, method=method)
        
        # 转换为图像
        real_rd = self.tensor_to_image(real_rd_tensor)
        heatmap_np = heatmap.squeeze().cpu().numpy()
        
        return real_rd, heatmap_np
    
    def visualize_results(self, sim_rd_path, real_rd, heatmap, save_path=None):
        """
        可视化结果
        
        Args:
            sim_rd_path: 仿真图路径
            real_rd: 生成的真实图
            heatmap: 热力图
            save_path: 保存路径（可选）
        """
        # 加载原始仿真图
        sim_rd_img = Image.open(sim_rd_path).convert('RGB')
        sim_rd_img = sim_rd_img.resize((self.config.img_size, self.config.img_size))
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 仿真图
        axes[0].imshow(sim_rd_img)
        axes[0].set_title('Sim RD', fontsize=14)
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title('Heatmap', fontsize=14)
        axes[1].axis('off')
        
        # 生成的真实图
        axes[2].imshow(real_rd)
        axes[2].set_title('Generated Real RD', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="推理Flow Matching RD图Sim2Real模型")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--sim_rd', type=str, required=True, help='仿真RD图路径')
    parser.add_argument('--prompt', type=str, default=None, help='文本描述（可选，不提供则自动检测目标）')
    parser.add_argument('--output', type=str, default='./output.png', help='输出路径')
    parser.add_argument('--steps', type=int, default=20, help='ODE求解步数')
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'], help='ODE求解方法')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = Inferencer(args.checkpoint, device=args.device)
    
    # 推理
    real_rd, heatmap = inferencer.inference(
        sim_rd_path=args.sim_rd,
        prompt=args.prompt,
        num_steps=args.steps,
        method=args.method
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(real_rd).save(output_path)
    print(f"真实RD图已保存: {output_path}")
    
    # 可视化
    if args.visualize:
        vis_path = output_path.parent / f"{output_path.stem}_vis.png"
        inferencer.visualize_results(args.sim_rd, real_rd, heatmap, save_path=vis_path)


if __name__ == "__main__":
    main()

