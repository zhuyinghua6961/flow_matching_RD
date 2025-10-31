"""
测试脚本 V2 - 在测试集上评估模型
计算定量指标：MSE, PSNR, SSIM
"""
import os
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models_v2 import Sim2RealFlowModel
from utils_v2 import RDPairDataset
from torch.utils.data import DataLoader


class Tester:
    def __init__(self, checkpoint_path, config_path='config_v2.yaml', device='cuda'):
        self.device = torch.device(device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        
        # 加载测试集
        self.test_loader = self._load_test_data()
        
        # 反归一化
        self.denormalize = transforms.Normalize(
            mean=[-self.config['data']['normalize_mean'] / self.config['data']['normalize_std']],
            std=[1.0 / self.config['data']['normalize_std']]
        )
        
        print("="*60)
        print("测试器已准备就绪")
        print(f"  检查点: {checkpoint_path}")
        print(f"  测试集大小: {len(self.test_loader.dataset)}")
        print(f"  设备: {self.device}")
        print("="*60)
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        print(f"加载模型: {checkpoint_path}")
        
        model_cfg = self.config['model']
        model = Sim2RealFlowModel(
            base_channels=model_cfg['base_channels'],
            channel_mult=tuple(model_cfg['channel_mult']),
            time_embed_dim=model_cfg['time_embed_dim'],
            num_res_blocks=model_cfg['num_res_blocks'],
            attention_levels=tuple(model_cfg['attention_levels']),
            dropout=model_cfg['dropout']
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"模型加载成功！(Epoch {checkpoint['epoch']})")
        
        return model
    
    def _load_test_data(self):
        """加载测试集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.config['data']['normalize_mean']],
                std=[self.config['data']['normalize_std']]
            )
        ])
        
        test_dataset = RDPairDataset(
            data_root=self.config['data']['test_root'],
            transform=transform,
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 逐张测试
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return test_loader
    
    def tensor_to_numpy(self, tensor):
        """将tensor转换为numpy数组"""
        # 反归一化
        img = self.denormalize(tensor.squeeze(0))
        img = img.squeeze(0).cpu().numpy()
        img = np.clip(img, 0, 1)
        return img
    
    def compute_metrics(self, generated, target):
        """
        计算评估指标
        
        Args:
            generated: numpy array (H, W) - 生成图像
            target: numpy array (H, W) - 目标图像
        
        Returns:
            metrics: dict - 评估指标
        """
        # MSE
        mse = np.mean((generated - target) ** 2)
        
        # PSNR
        psnr_value = psnr(target, generated, data_range=1.0)
        
        # SSIM
        ssim_value = ssim(target, generated, data_range=1.0)
        
        return {
            'mse': mse,
            'psnr': psnr_value,
            'ssim': ssim_value
        }
    
    @torch.no_grad()
    def test(self, save_results=False, output_dir=None):
        """
        在测试集上评估
        
        Args:
            save_results: bool - 是否保存生成结果
            output_dir: str - 输出目录
        
        Returns:
            avg_metrics: dict - 平均指标
        """
        if save_results:
            output_dir = Path(output_dir) if output_dir else Path(self.config['paths']['output_dir']) / 'test_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"保存结果到: {output_dir}")
        
        all_metrics = []
        
        ode_steps = self.config['inference']['ode_steps']
        ode_method = self.config['inference']['ode_method']
        
        print(f"\n开始测试...")
        
        for sim_images, real_images, names in tqdm(self.test_loader, desc="Testing"):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # 生成
            generated = self.model.generate(sim_images, ode_steps=ode_steps, ode_method=ode_method)
            
            # 转换为numpy
            generated_np = self.tensor_to_numpy(generated)
            real_np = self.tensor_to_numpy(real_images)
            
            # 计算指标
            metrics = self.compute_metrics(generated_np, real_np)
            all_metrics.append(metrics)
            
            # 保存结果
            if save_results:
                name = names[0]
                
                # 保存生成图
                gen_path = output_dir / f"{name}_generated.png"
                Image.fromarray((generated_np * 255).astype(np.uint8), mode='L').save(gen_path)
                
                # 保存对比图（并排显示）
                sim_np = self.tensor_to_numpy(sim_images)
                comparison = np.hstack([sim_np, generated_np, real_np])
                comp_path = output_dir / f"{name}_comparison.png"
                Image.fromarray((comparison * 255).astype(np.uint8), mode='L').save(comp_path)
        
        # 计算平均指标
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in all_metrics]),
            'psnr': np.mean([m['psnr'] for m in all_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_metrics])
        }
        
        # 标准差
        std_metrics = {
            'mse_std': np.std([m['mse'] for m in all_metrics]),
            'psnr_std': np.std([m['psnr'] for m in all_metrics]),
            'ssim_std': np.std([m['ssim'] for m in all_metrics])
        }
        
        # 打印结果
        print("\n" + "="*60)
        print("测试结果:")
        print("="*60)
        print(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse_std']:.6f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB ± {std_metrics['psnr_std']:.2f}")
        print(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}")
        print("="*60)
        
        # 保存指标到文件
        if save_results:
            metrics_file = output_dir / "metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write("测试集评估指标\n")
                f.write("="*60 + "\n")
                f.write(f"测试样本数: {len(all_metrics)}\n")
                f.write(f"ODE步数: {ode_steps}\n")
                f.write(f"ODE方法: {ode_method}\n\n")
                f.write("平均指标:\n")
                f.write(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse_std']:.6f}\n")
                f.write(f"  PSNR: {avg_metrics['psnr']:.2f} dB ± {std_metrics['psnr_std']:.2f}\n")
                f.write(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}\n")
            print(f"\n指标已保存到: {metrics_file}")
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Test Sim2Real Flow Matching V2")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备：cuda或cpu')
    parser.add_argument('--save_results', action='store_true', help='保存生成结果')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    tester = Tester(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    tester.test(save_results=args.save_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

