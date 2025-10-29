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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
            controlnet_base_channels=32,
            controlnet_channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_levels=(2, 3),
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
    def inference(self, sim_rd_path, prompt, num_steps=20, method='euler'):
        """
        推理生成真实RD图
        
        Args:
            sim_rd_path: 仿真RD图路径
            prompt: 文本描述或参数字典
            num_steps: ODE求解步数
            method: 'euler' 或 'rk4'
        
        Returns:
            real_rd: 生成的真实RD图 (numpy array)
            heatmap: 热力图 (numpy array)
        """
        # 加载仿真图
        sim_rd = self.load_image(sim_rd_path).to(self.device)
        
        # 生成热力图
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
    parser.add_argument('--prompt', type=str, required=True, help='文本描述，如"速度: 5m/s, 距离: 100m"')
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

