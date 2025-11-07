"""
测试脚本 V3 - 在测试集上评估模型
计算定量指标和综合评分
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

from models_v2 import Sim2RealFlowModel
from utils_v2 import RDPairDataset
from utils_v3 import ImageQualityMetrics, ModelEvaluator
from torch.utils.data import DataLoader


class TesterV3:
    def __init__(self, checkpoint_path, config_path='config_v2.yaml', device='cuda'):
        self.device = torch.device(device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model, self.checkpoint = self._load_model(checkpoint_path)
        
        # 加载测试集
        self.test_loader = self._load_test_data()
        
        # 反归一化
        self.denormalize = transforms.Normalize(
            mean=[-self.config['data']['normalize_mean'] / self.config['data']['normalize_std']],
            std=[1.0 / self.config['data']['normalize_std']]
        )
        
        # 创建评估器（使用检查点中的PSNR范围，如果有）
        psnr_range = self.checkpoint.get('psnr_range', None)
        self.evaluator = ModelEvaluator(psnr_range=psnr_range)
        
        print("="*60)
        print("测试器 V3 已准备就绪")
        print(f"  检查点: {checkpoint_path}")
        print(f"  测试集大小: {len(self.test_loader.dataset)}")
        print(f"  设备: {self.device}")
        if psnr_range:
            print(f"  PSNR范围: [{psnr_range['min']:.1f}, {psnr_range['max']:.1f}] dB")
        else:
            print(f"  PSNR范围: 使用默认 [15.0, 35.0] dB")
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
        
        return model, checkpoint
    
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
    
    @torch.no_grad()
    def test(self, save_results=False, output_dir=None):
        """
        在测试集上评估
        
        Args:
            save_results: bool - 是否保存生成结果
            output_dir: str - 输出目录
        
        Returns:
            result: dict - 评估结果
                - avg_metrics: dict - 平均指标
                - score: dict - 综合评分
                - all_metrics: list - 所有样本的指标
        """
        if save_results:
            output_dir = Path(output_dir) if output_dir else Path(self.config['paths']['output_dir']) / 'test_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"保存结果到: {output_dir}")
        
        all_metrics = []
        
        ode_steps = self.config['inference']['ode_steps']
        ode_method = self.config['inference']['ode_method']
        
        print(f"\n开始测试...")
        print(f"  ODE步数: {ode_steps}")
        print(f"  ODE方法: {ode_method}")
        
        for sim_images, real_images, names in tqdm(self.test_loader, desc="Testing"):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # 生成
            generated = self.model.generate(sim_images, ode_steps=ode_steps, ode_method=ode_method)
            
            # 转换为numpy
            generated_np = self.tensor_to_numpy(generated)
            real_np = self.tensor_to_numpy(real_images)
            
            # 计算所有指标
            metrics = ImageQualityMetrics.compute_all_metrics(
                generated_np, real_np, include_frequency=True
            )
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
        metric_keys = all_metrics[0].keys()
        avg_metrics = {}
        std_metrics = {}
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            # 过滤inf值
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                avg_metrics[key] = np.mean(finite_values)
                std_metrics[f'{key}_std'] = np.std(finite_values)
            else:
                avg_metrics[key] = 0.0
                std_metrics[f'{key}_std'] = 0.0
        
        # 计算综合评分
        score = self.evaluator.calculate_score(avg_metrics)
        
        # 打印结果
        self._print_results(avg_metrics, std_metrics, score, len(all_metrics), ode_steps, ode_method)
        
        # 保存结果
        if save_results:
            self._save_results(avg_metrics, std_metrics, score, all_metrics, output_dir, ode_steps, ode_method)
        
        return {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'score': score,
            'all_metrics': all_metrics
        }
    
    def _print_results(self, avg_metrics, std_metrics, score, num_samples, ode_steps, ode_method):
        """打印结果"""
        print("\n" + "="*60)
        print("测试结果")
        print("="*60)
        print(f"测试样本数: {num_samples}")
        print(f"ODE步数: {ode_steps}")
        print(f"ODE方法: {ode_method}")
        
        # 基础指标
        print("\n【基础指标】")
        print(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse_std']:.6f}")
        print(f"  MAE:  {avg_metrics['mae']:.6f} ± {std_metrics['mae_std']:.6f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB ± {std_metrics['psnr_std']:.2f}")
        print(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}")
        
        # 频域指标
        print("\n【频域指标】⭐⭐⭐（核心指标）")
        print(f"  Freq MSE:        {avg_metrics['freq_mse']:.6f} ± {std_metrics['freq_mse_std']:.6f}")
        if np.isfinite(avg_metrics['freq_psnr']):
            print(f"  Freq PSNR:       {avg_metrics['freq_psnr']:.2f} dB ± {std_metrics['freq_psnr_std']:.2f}")
        else:
            print(f"  Freq PSNR:       inf")
        print(f"  Freq Correlation: {avg_metrics['freq_correlation']:.4f} ± {std_metrics['freq_correlation_std']:.4f}")
        
        # 其他指标
        print("\n【其他指标】")
        print(f"  Energy Similarity: {avg_metrics['energy_similarity']:.4f} ± {std_metrics['energy_similarity_std']:.4f}")
        
        # 综合评分
        print("\n" + self.evaluator.format_score_report(score, avg_metrics))
    
    def _save_results(self, avg_metrics, std_metrics, score, all_metrics, output_dir, ode_steps, ode_method):
        """保存结果到文件"""
        # 保存指标到文件
        metrics_file = output_dir / "metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("测试集评估指标\n")
            f.write("="*60 + "\n")
            f.write(f"测试样本数: {len(all_metrics)}\n")
            f.write(f"ODE步数: {ode_steps}\n")
            f.write(f"ODE方法: {ode_method}\n\n")
            
            # 基础指标
            f.write("【基础指标】\n")
            f.write(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse_std']:.6f}\n")
            f.write(f"  MAE:  {avg_metrics['mae']:.6f} ± {std_metrics['mae_std']:.6f}\n")
            f.write(f"  PSNR: {avg_metrics['psnr']:.2f} dB ± {std_metrics['psnr_std']:.2f}\n")
            f.write(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}\n\n")
            
            # 频域指标
            f.write("【频域指标】⭐⭐⭐（核心指标）\n")
            f.write(f"  Freq MSE:        {avg_metrics['freq_mse']:.6f} ± {std_metrics['freq_mse_std']:.6f}\n")
            if np.isfinite(avg_metrics['freq_psnr']):
                f.write(f"  Freq PSNR:       {avg_metrics['freq_psnr']:.2f} dB ± {std_metrics['freq_psnr_std']:.2f}\n")
            else:
                f.write(f"  Freq PSNR:       inf\n")
            f.write(f"  Freq Correlation: {avg_metrics['freq_correlation']:.4f} ± {std_metrics['freq_correlation_std']:.4f}\n\n")
            
            # 其他指标
            f.write("【其他指标】\n")
            f.write(f"  Energy Similarity: {avg_metrics['energy_similarity']:.4f} ± {std_metrics['energy_similarity_std']:.4f}\n\n")
            
            # 综合评分
            f.write(self.evaluator.format_score_report(score, avg_metrics))
            f.write("\n")
        
        print(f"\n指标已保存到: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Test Sim2Real Flow Matching V3")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备：cuda或cpu')
    parser.add_argument('--save_results', action='store_true', help='保存生成结果')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    tester = TesterV3(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    tester.test(save_results=args.save_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

