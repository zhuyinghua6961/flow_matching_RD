"""
推理脚本 V2 - 从仿真图生成真实图
无需prompt，直接输入仿真图即可
"""
import os
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models_v2 import Sim2RealFlowModel


class Sim2RealInference:
    def __init__(self, checkpoint_path, config_path='config_v2.yaml', device='cuda'):
        """
        初始化推理器
        
        Args:
            checkpoint_path: str - 模型检查点路径
            config_path: str - 配置文件路径
            device: str - 设备
        """
        self.device = torch.device(device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.config['data']['normalize_mean']],
                std=[self.config['data']['normalize_std']]
            )
        ])
        
        # 反归一化
        self.denormalize = transforms.Normalize(
            mean=[-self.config['data']['normalize_mean'] / self.config['data']['normalize_std']],
            std=[1.0 / self.config['data']['normalize_std']]
        )
        
        print("="*60)
        print("推理器已准备就绪")
        print(f"  检查点: {checkpoint_path}")
        print(f"  设备: {self.device}")
        print(f"  ODE步数: {self.config['inference']['ode_steps']}")
        print(f"  ODE方法: {self.config['inference']['ode_method']}")
        print("="*60)
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        print(f"加载模型: {checkpoint_path}")
        
        # 创建模型
        model_cfg = self.config['model']
        model = Sim2RealFlowModel(
            base_channels=model_cfg['base_channels'],
            channel_mult=tuple(model_cfg['channel_mult']),
            time_embed_dim=model_cfg['time_embed_dim'],
            num_res_blocks=model_cfg['num_res_blocks'],
            attention_levels=tuple(model_cfg['attention_levels']),
            dropout=model_cfg['dropout']
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"模型加载成功！(Epoch {checkpoint['epoch']})")
        
        return model
    
    def load_image(self, image_path):
        """
        加载图像
        
        Args:
            image_path: str - 图像路径
        Returns:
            image_tensor: (1, 1, H, W) - 图像tensor
        """
        img = Image.open(image_path).convert('L')  # 灰度图
        img_tensor = self.transform(img).unsqueeze(0)  # (1, 1, H, W)
        return img_tensor.to(self.device)
    
    def save_image(self, image_tensor, save_path):
        """
        保存图像
        
        Args:
            image_tensor: (1, 1, H, W) - 图像tensor
            save_path: str - 保存路径
        """
        # 反归一化
        img = self.denormalize(image_tensor.squeeze(0))  # (1, H, W)
        img = img.squeeze(0).cpu().numpy()  # (H, W)
        
        # 裁剪到[0, 1]
        img = np.clip(img, 0, 1)
        
        # 转换为PIL图像
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        
        # 保存
        img.save(save_path)
        print(f"保存图像: {save_path}")
    
    @torch.no_grad()
    def inference(self, sim_image_path, save_path=None, save_intermediate=False):
        """
        推理 - 从仿真图生成真实图
        
        Args:
            sim_image_path: str - 仿真图路径
            save_path: str - 保存路径（None=自动生成）
            save_intermediate: bool - 是否保存中间步骤
        
        Returns:
            generated_image: PIL.Image - 生成的真实图
        """
        # 加载仿真图
        sim_image = self.load_image(sim_image_path)
        
        # 生成真实图
        print(f"\n生成真实图...")
        print(f"  输入: {sim_image_path}")
        print(f"  ODE步数: {self.config['inference']['ode_steps']}")
        
        ode_steps = self.config['inference']['ode_steps']
        ode_method = self.config['inference']['ode_method']
        
        if save_intermediate and self.config['inference']['save_intermediate']:
            # 保存中间步骤
            generated = self._generate_with_intermediate(sim_image, ode_steps, ode_method, save_path)
        else:
            # 直接生成
            generated = self.model.generate(sim_image, ode_steps=ode_steps, ode_method=ode_method)
        
        # 保存结果
        if save_path is None:
            # 自动生成路径
            input_path = Path(sim_image_path)
            save_path = input_path.parent / f"{input_path.stem}_generated.png"
        
        self.save_image(generated, save_path)
        
        # 转换为PIL图像返回
        img = self.denormalize(generated.squeeze(0)).squeeze(0).cpu().numpy()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        generated_pil = Image.fromarray(img, mode='L')
        
        print(f"生成完成！")
        
        return generated_pil
    
    def _generate_with_intermediate(self, sim_image, ode_steps, ode_method, save_path):
        """生成并保存中间步骤"""
        device = sim_image.device
        batch_size = sim_image.shape[0]
        
        # 初始化
        x = torch.randn_like(sim_image)
        dt = 1.0 / ode_steps
        
        # 保存目录
        save_dir = Path(save_path).parent / f"{Path(save_path).stem}_steps"
        save_dir.mkdir(exist_ok=True)
        
        intermediate_steps = self.config['inference']['intermediate_steps']
        
        for i in range(ode_steps):
            t = torch.ones(batch_size, device=device) * (i * dt)
            v = self.model(x, t, sim_image)
            x = x + v * dt
            
            # 保存中间步骤
            if i in intermediate_steps:
                step_path = save_dir / f"step_{i:03d}.png"
                self.save_image(x, step_path)
        
        return x
    
    def batch_inference(self, input_dir, output_dir):
        """
        批量推理
        
        Args:
            input_dir: str - 输入目录（包含仿真图）
            output_dir: str - 输出目录
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像
        image_files = list(input_dir.glob("*.png"))
        
        print(f"\n批量推理:")
        print(f"  输入目录: {input_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  图像数量: {len(image_files)}")
        
        for img_file in image_files:
            save_path = output_dir / f"{img_file.stem}_generated.png"
            try:
                self.inference(str(img_file), str(save_path))
            except Exception as e:
                print(f"处理失败 {img_file}: {e}")
        
        print(f"\n批量推理完成！")


def main():
    parser = argparse.ArgumentParser(description="Sim2Real Inference V2")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入仿真图路径或目录')
    parser.add_argument('--output', type=str, default=None, help='输出路径或目录')
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备：cuda或cpu')
    parser.add_argument('--save_intermediate', action='store_true', help='保存中间步骤')
    parser.add_argument('--batch', action='store_true', help='批量推理模式')
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = Sim2RealInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 推理
    if args.batch:
        # 批量推理
        output_dir = args.output if args.output else Path(args.input).parent / "generated"
        inferencer.batch_inference(args.input, output_dir)
    else:
        # 单张推理
        inferencer.inference(
            sim_image_path=args.input,
            save_path=args.output,
            save_intermediate=args.save_intermediate
        )


if __name__ == "__main__":
    main()

