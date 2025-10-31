"""
测试脚本 - 在测试集上评估模型性能
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import json

from models import FlowMatchingModel
from utils import HeatmapGenerator, create_dataloader
from config import load_config
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class Tester:
    """测试器"""
    
    def __init__(self, checkpoint_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        print(f"使用设备: {self.device}")
        print(f"加载检查点: {checkpoint_path}")
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = FlowMatchingModel(
            unet_base_channels=64,
            unet_channel_mult=(1, 2, 4, 8),
            controlnet_base_channels=64,
            controlnet_channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_levels=(3,),
            dropout=0.0,
            time_emb_dim=256
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ 模型加载完成")
        
        # 热力图生成器
        self.heatmap_generator = HeatmapGenerator(
            img_size=config.img_size,
            max_speed=config.max_speed,
            max_range=config.max_range,
            sigma=config.heatmap_sigma
        )
    
    def tensor_to_numpy(self, tensor):
        """将tensor转换为numpy [0, 255]"""
        # 反归一化: [-1, 1] -> [0, 1]
        img = (tensor + 1) / 2
        img = img.clamp(0, 1)
        img = img.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        img = (img * 255).astype(np.uint8)
        return img
    
    def compute_metrics(self, pred, target):
        """
        计算评估指标
        
        Args:
            pred: 预测图像 (H, W, C) uint8
            target: 目标图像 (H, W, C) uint8
        
        Returns:
            dict: 包含各种指标的字典
        """
        # MSE
        mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
        
        # PSNR
        psnr = peak_signal_noise_ratio(target, pred, data_range=255)
        
        # SSIM
        ssim = structural_similarity(target, pred, multichannel=True, channel_axis=2, data_range=255)
        
        # MAE
        mae = np.mean(np.abs(pred.astype(float) - target.astype(float)))
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'mae': float(mae)
        }
    
    @torch.no_grad()
    def test(self, test_loader, num_steps=20, method='euler', save_results=False, output_dir=None):
        """
        在测试集上评估
        
        Args:
            test_loader: 测试数据加载器
            num_steps: ODE求解步数
            method: ODE求解方法
            save_results: 是否保存推理结果
            output_dir: 保存目录
        
        Returns:
            dict: 测试结果统计
        """
        all_metrics = {
            'mse': [],
            'psnr': [],
            'ssim': [],
            'mae': []
        }
        
        if save_results:
            output_dir = Path(output_dir or './test_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"结果将保存到: {output_dir}")
        
        print(f"\n开始测试... (样本数: {len(test_loader.dataset)})")
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            sim_rd = batch['sim_rd'].to(self.device)
            real_rd_gt = batch['real_rd'].to(self.device)
            prompts = batch['prompt']
            filenames = batch['filename']
            
            # 生成热力图
            heatmaps = []
            for prompt in prompts:
                heatmap = self.heatmap_generator(prompt)
                heatmaps.append(heatmap)
            heatmap = torch.stack(heatmaps).to(self.device)
            
            # 推理生成
            real_rd_pred = self.model.sample(sim_rd, heatmap, num_steps=num_steps, method=method)
            
            # 逐样本评估
            batch_size = sim_rd.shape[0]
            for i in range(batch_size):
                # 转换为numpy
                pred_np = self.tensor_to_numpy(real_rd_pred[i])
                target_np = self.tensor_to_numpy(real_rd_gt[i])
                
                # 计算指标
                metrics = self.compute_metrics(pred_np, target_np)
                
                for key in all_metrics.keys():
                    all_metrics[key].append(metrics[key])
                
                # 保存结果（可选）
                if save_results:
                    filename = filenames[i]
                    
                    # 保存预测图
                    Image.fromarray(pred_np).save(output_dir / f"{filename}_pred.png")
                    
                    # 保存对比图
                    sim_np = self.tensor_to_numpy(sim_rd[i])
                    
                    # 拼接三张图
                    comparison = np.concatenate([sim_np, pred_np, target_np], axis=1)
                    Image.fromarray(comparison).save(output_dir / f"{filename}_comparison.png")
        
        # 计算统计
        stats = {}
        for key in all_metrics.keys():
            values = all_metrics[key]
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats, all_metrics


def main():
    parser = argparse.ArgumentParser(description="测试Flow Matching RD图Sim2Real模型")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--test_data', type=str, default='./dataset/test', help='测试集路径')
    parser.add_argument('--steps', type=int, default=20, help='ODE求解步数')
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'], help='ODE求解方法')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--save_results', action='store_true', help='保存测试结果图像')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建测试数据加载器
    print(f"\n加载测试集: {args.test_data}")
    test_loader = create_dataloader(
        data_root=args.test_data,
        batch_size=args.batch_size,
        img_size=config.img_size,
        num_workers=4,
        shuffle=False,
        split='test'
    )
    
    # 创建测试器
    tester = Tester(args.checkpoint, config, device=args.device)
    
    # 测试
    stats, all_metrics = tester.test(
        test_loader=test_loader,
        num_steps=args.steps,
        method=args.method,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果统计")
    print("="*60)
    
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:   {metric_stats['mean']:.4f}")
        print(f"  Std:    {metric_stats['std']:.4f}")
        print(f"  Min:    {metric_stats['min']:.4f}")
        print(f"  Max:    {metric_stats['max']:.4f}")
        print(f"  Median: {metric_stats['median']:.4f}")
    
    # 保存统计结果
    if args.save_results:
        output_dir = Path(args.output_dir)
        stats_file = output_dir / 'test_statistics.json'
        
        with open(stats_file, 'w') as f:
            json.dump({
                'stats': stats,
                'all_values': all_metrics,
                'config': {
                    'checkpoint': args.checkpoint,
                    'test_data': args.test_data,
                    'num_steps': args.steps,
                    'method': args.method
                }
            }, f, indent=2)
        
        print(f"\n✓ 统计结果已保存: {stats_file}")
    
    print("="*60)


if __name__ == "__main__":
    main()

