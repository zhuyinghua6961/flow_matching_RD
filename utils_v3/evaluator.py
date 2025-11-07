"""
模型评估器 V3
提供综合评分和评估功能
"""
import torch
import numpy as np
from .metrics import ImageQualityMetrics


class ModelEvaluator:
    """
    模型评估器
    提供综合评分和评估功能
    """
    
    def __init__(self, psnr_range=None):
        """
        初始化评估器
        
        Args:
            psnr_range: dict - PSNR范围 {'min': float, 'max': float}
                       如果为None，使用默认范围 [15, 35]
        """
        if psnr_range is None:
            self.psnr_min = 15.0
            self.psnr_max = 35.0
        else:
            self.psnr_min = psnr_range.get('min', 15.0)
            self.psnr_max = psnr_range.get('max', 35.0)
    
    def normalize_psnr(self, psnr_value):
        """
        归一化PSNR到[0, 1]
        
        Args:
            psnr_value: float - PSNR值
        
        Returns:
            normalized: float - 归一化后的值 [0, 1]
        """
        normalized = (psnr_value - self.psnr_min) / (self.psnr_max - self.psnr_min)
        return np.clip(normalized, 0, 1)
    
    def calculate_score(self, metrics):
        """
        计算综合评分 (0-100分)
        
        公式:
            分数 = 60 × 频域得分 + 25 × SSIM得分 + 15 × PSNR得分
        
        Args:
            metrics: dict - 评估指标字典
                - freq_correlation: float - 频域相关系数 [-1, 1]
                - ssim: float - SSIM值 [0, 1]
                - psnr: float - PSNR值 (dB)
        
        Returns:
            score: dict - 评分结果
                - total_score: float - 总分 (0-100)
                - freq_score: float - 频域得分 (0-60)
                - ssim_score: float - SSIM得分 (0-25)
                - psnr_score: float - PSNR得分 (0-15)
                - rating: str - 评级
        """
        # 1. 频域得分 (60分)
        freq_correlation = metrics.get('freq_correlation', 0.0)
        freq_score_normalized = (freq_correlation + 1) / 2  # 归一化到[0, 1]
        freq_score = 60 * np.clip(freq_score_normalized, 0, 1)
        
        # 2. SSIM得分 (25分)
        ssim_value = metrics.get('ssim', 0.0)
        ssim_score = 25 * np.clip(ssim_value, 0, 1)
        
        # 3. PSNR得分 (15分)
        psnr_value = metrics.get('psnr', 0.0)
        psnr_normalized = self.normalize_psnr(psnr_value)
        psnr_score = 15 * psnr_normalized
        
        # 4. 总分
        total_score = freq_score + ssim_score + psnr_score
        
        # 5. 评级
        rating = self._get_rating(total_score)
        
        return {
            'total_score': float(total_score),
            'freq_score': float(freq_score),
            'ssim_score': float(ssim_score),
            'psnr_score': float(psnr_score),
            'rating': rating,
            'freq_correlation': float(freq_correlation),
            'ssim': float(ssim_value),
            'psnr': float(psnr_value)
        }
    
    def _get_rating(self, score):
        """
        根据分数获取评级
        
        Args:
            score: float - 总分 (0-100)
        
        Returns:
            rating: str - 评级
        """
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "一般"
        elif score >= 60:
            return "需改进"
        else:
            return "较差"
    
    def evaluate_batch(self, generated_images, target_images, device='cpu'):
        """
        评估一批图像
        
        Args:
            generated_images: torch.Tensor (B, C, H, W) - 生成的图像
            target_images: torch.Tensor (B, C, H, W) - 目标图像
            device: str - 设备
        
        Returns:
            metrics: dict - 所有指标的平均值
        """
        batch_size = generated_images.shape[0]
        all_metrics = []
        
        for i in range(batch_size):
            gen_img = generated_images[i]
            tgt_img = target_images[i]
            
            # 转换为numpy
            gen_np = gen_img.cpu().numpy() if isinstance(gen_img, torch.Tensor) else gen_img
            tgt_np = tgt_img.cpu().numpy() if isinstance(tgt_img, torch.Tensor) else tgt_img
            
            # 计算指标
            metrics = ImageQualityMetrics.compute_all_metrics(
                gen_np, tgt_np, include_frequency=True
            )
            all_metrics.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            # 处理inf值
            values = [v for v in values if np.isfinite(v)]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def format_score_report(self, score_dict, metrics_dict=None):
        """
        格式化评分报告
        
        Args:
            score_dict: dict - 评分结果
            metrics_dict: dict - 完整指标字典（可选）
        
        Returns:
            report: str - 格式化的报告字符串
        """
        lines = []
        lines.append("="*60)
        lines.append("综合评分报告")
        lines.append("="*60)
        lines.append(f"总分: {score_dict['total_score']:.2f} / 100")
        lines.append(f"评级: {score_dict['rating']}")
        lines.append("")
        lines.append("评分明细:")
        lines.append(f"  频域得分: {score_dict['freq_score']:.2f} / 60")
        lines.append(f"    - 频域相关系数: {score_dict['freq_correlation']:.4f}")
        lines.append(f"  SSIM得分: {score_dict['ssim_score']:.2f} / 25")
        lines.append(f"    - SSIM值: {score_dict['ssim']:.4f}")
        lines.append(f"  PSNR得分: {score_dict['psnr_score']:.2f} / 15")
        lines.append(f"    - PSNR值: {score_dict['psnr']:.2f} dB")
        lines.append(f"    - PSNR范围: [{self.psnr_min:.1f}, {self.psnr_max:.1f}] dB")
        
        if metrics_dict:
            lines.append("")
            lines.append("其他指标:")
            if 'mse' in metrics_dict:
                lines.append(f"  MSE: {metrics_dict['mse']:.6f}")
            if 'mae' in metrics_dict:
                lines.append(f"  MAE: {metrics_dict['mae']:.6f}")
            if 'freq_mse' in metrics_dict:
                lines.append(f"  频域MSE: {metrics_dict['freq_mse']:.6f}")
            if 'energy_similarity' in metrics_dict:
                lines.append(f"  能量分布相似度: {metrics_dict['energy_similarity']:.4f}")
        
        lines.append("="*60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("ModelEvaluator V3 测试")
    print("="*60)
    
    # 创建评估器
    evaluator = ModelEvaluator(psnr_range={'min': 15.0, 'max': 35.0})
    
    # 模拟指标
    metrics = {
        'freq_correlation': 0.82,
        'ssim': 0.88,
        'psnr': 29.5,
        'mse': 0.0012,
        'mae': 0.012,
        'freq_mse': 0.0056,
        'energy_similarity': 0.91
    }
    
    # 计算评分
    score = evaluator.calculate_score(metrics)
    
    # 格式化报告
    report = evaluator.format_score_report(score, metrics)
    print("\n" + report)

