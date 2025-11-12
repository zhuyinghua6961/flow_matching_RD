"""
V4 微调模型测试脚本
测试微调后的模型效果
"""
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models_v2 import Sim2RealFlowModel
from utils_v2 import RDPairDataset
from utils_v3 import ImageQualityMetrics, ModelEvaluator


class FineTunedTester:
    """微调模型测试器"""
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        self.device = torch.device(device)
        
        # 加载配置
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model, self.checkpoint = self._load_model(checkpoint_path)
        
        # 反归一化
        self.denormalize = transforms.Normalize(
            mean=[-self.config['data']['normalize_mean'] / self.config['data']['normalize_std']],
            std=[1.0 / self.config['data']['normalize_std']]
        )
        
        # 创建评估器
        psnr_range = self.checkpoint.get('psnr_range', None)
        self.evaluator = ModelEvaluator(psnr_range=psnr_range)
        
        print("="*60)
        print("V4 微调模型测试器")
        print(f"检查点: {checkpoint_path}")
        print(f"Epoch: {self.checkpoint['epoch']}")
        if psnr_range:
            print(f"PSNR范围: [{psnr_range['min']:.1f}, {psnr_range['max']:.1f}] dB")
        print("="*60)
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
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
        
        return model, checkpoint
    
    def test(self, test_loader, save_results=False, output_dir=None):
        """测试"""
        if save_results:
            output_dir = Path(output_dir) if output_dir else Path('./v4_test_results')
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        
        print("\n开始测试...")
        for sim_images, real_images, names in tqdm(test_loader, desc="Testing"):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # 生成
            generated = self.model.generate(
                sim_images,
                ode_steps=50,
                ode_method='euler'
            )
            
            # 转换为numpy
            generated_np = self.tensor_to_numpy(generated)
            real_np = self.tensor_to_numpy(real_images)
            
            # 计算指标
            metrics = ImageQualityMetrics.compute_all_metrics(
                generated_np, real_np, include_frequency=True
            )
            all_metrics.append(metrics)
            
            # 保存结果
            if save_results:
                name = names[0]
                gen_path = output_dir / f"{name}_generated.png"
                Image.fromarray((generated_np * 255).astype(np.uint8), mode='L').save(gen_path)
                
                # 对比图
                sim_np = self.tensor_to_numpy(sim_images)
                comparison = np.hstack([sim_np, generated_np, real_np])
                comp_path = output_dir / f"{name}_comparison.png"
                Image.fromarray((comparison * 255).astype(np.uint8), mode='L').save(comp_path)
        
        # 计算平均指标
        avg_metrics = self._compute_average_metrics(all_metrics)
        
        # 计算综合评分
        score = self.evaluator.calculate_score(avg_metrics)
        
        # 打印结果
        self._print_results(avg_metrics, score, len(all_metrics))
        
        return {'avg_metrics': avg_metrics, 'score': score}
    
    def tensor_to_numpy(self, tensor):
        """tensor转numpy"""
        img = self.denormalize(tensor.squeeze(0))
        img = img.squeeze(0).cpu().numpy()
        img = np.clip(img, 0, 1)
        return img
    
    def _compute_average_metrics(self, all_metrics):
        """计算平均指标"""
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                avg_metrics[key] = np.mean(finite_values)
            else:
                avg_metrics[key] = 0.0
        return avg_metrics
    
    def _print_results(self, avg_metrics, score, num_samples):
        """打印结果"""
        print("\n" + "="*60)
        print("V4 微调模型测试结果")
        print("="*60)
        print(f"测试样本数: {num_samples}")
        
        print("\n【基础指标】")
        print(f"  MSE:  {avg_metrics['mse']:.6f}")
        print(f"  MAE:  {avg_metrics['mae']:.6f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {avg_metrics['ssim']:.4f}")
        
        print("\n【频域指标】⭐⭐⭐（多普勒效应）")
        print(f"  Freq MSE:        {avg_metrics['freq_mse']:.6f}")
        if np.isfinite(avg_metrics['freq_psnr']):
            print(f"  Freq PSNR:       {avg_metrics['freq_psnr']:.2f} dB")
        print(f"  Freq Correlation: {avg_metrics['freq_correlation']:.4f}")
        
        print("\n" + self.evaluator.format_score_report(score, avg_metrics))


def main():
    parser = argparse.ArgumentParser(description="Test V4 Fine-tuned Model")
    parser.add_argument('--checkpoint', type=str, default=None, help='微调模型检查点（可选，默认使用配置文件中的final_finetuned.pth）')
    parser.add_argument('--config', type=str, default='v4_finetune/config_finetune.yaml', help='配置文件')
    parser.add_argument('--test_root', type=str, default=None, help='测试集路径（可选，默认使用配置文件）')
    parser.add_argument('--save_results', action='store_true', help='保存生成结果')
    parser.add_argument('--output_dir', type=str, default='./v4_test_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    # 读取配置文件
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 默认使用final_finetuned.pth
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_path = str(checkpoint_dir / 'final_finetuned.pth')
        print(f"使用配置文件中的模型路径: {checkpoint_path}")
    else:
        print(f"使用命令行指定的模型: {checkpoint_path}")
    
    # 确定测试集路径
    test_root = args.test_root
    if test_root is None:
        test_root = config['data']['test_root']
    
    # 创建测试器
    tester = FineTunedTester(checkpoint_path, args.config, args.device)
    
    # 加载测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[config['data']['normalize_mean']],
            std=[config['data']['normalize_std']]
        )
    ])
    
    test_dataset = RDPairDataset(
        data_root=test_root,
        transform=transform,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 开始测试
    tester.test(test_loader, args.save_results, args.output_dir)


if __name__ == "__main__":
    main()
