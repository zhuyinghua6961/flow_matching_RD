"""
图像质量评估指标模块 V3
用于评估Sim2Real生成的雷达RD图质量
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageQualityMetrics:
    """
    图像质量评估指标集合
    支持PyTorch tensor和numpy array输入
    """
    
    @staticmethod
    def _to_numpy(tensor_or_array):
        """转换为numpy数组"""
        if isinstance(tensor_or_array, torch.Tensor):
            if tensor_or_array.is_cuda:
                tensor_or_array = tensor_or_array.cpu()
            return tensor_or_array.numpy()
        return tensor_or_array
    
    @staticmethod
    def _to_tensor(array_or_tensor, device='cpu'):
        """转换为PyTorch tensor"""
        if isinstance(array_or_tensor, np.ndarray):
            return torch.from_numpy(array_or_tensor).float().to(device)
        return array_or_tensor.to(device)
    
    @staticmethod
    def _ensure_4d(img):
        """确保图像是4D tensor (B, C, H, W)"""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img = img.unsqueeze(0)
            else:
                img = img.unsqueeze(1)
        
        return img
    
    @staticmethod
    def mse(pred, target):
        """均方误差"""
        if isinstance(pred, torch.Tensor):
            return F.mse_loss(pred, target).item()
        else:
            return np.mean((pred - target) ** 2)
    
    @staticmethod
    def mae(pred, target):
        """平均绝对误差"""
        if isinstance(pred, torch.Tensor):
            return torch.mean(torch.abs(pred - target)).item()
        else:
            return np.mean(np.abs(pred - target))
    
    @staticmethod
    def psnr_score(pred, target, data_range=1.0):
        """峰值信噪比"""
        pred_np = ImageQualityMetrics._to_numpy(pred)
        target_np = ImageQualityMetrics._to_numpy(target)
        return psnr(target_np, pred_np, data_range=data_range)
    
    @staticmethod
    def ssim_score(pred, target, data_range=1.0):
        """结构相似性指数"""
        pred_np = ImageQualityMetrics._to_numpy(pred)
        target_np = ImageQualityMetrics._to_numpy(target)
        # 确保是2D数组
        if pred_np.ndim > 2:
            pred_np = pred_np.squeeze()
            target_np = target_np.squeeze()
        return ssim(target_np, pred_np, data_range=data_range)
    
    @staticmethod
    def frequency_spectrum_mse(pred, target):
        """
        频域幅度谱MSE
        
        Args:
            pred: (H, W) or (B, C, H, W) - 预测图像
            target: (H, W) or (B, C, H, W) - 目标图像
        
        Returns:
            freq_mse: float - 频域幅度谱MSE
        """
        pred_t = ImageQualityMetrics._ensure_4d(pred)
        target_t = ImageQualityMetrics._ensure_4d(target)
        
        # FFT变换到频域
        pred_fft = torch.fft.rfft2(pred_t, norm='ortho')
        target_fft = torch.fft.rfft2(target_t, norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 对数域（更稳定）
        pred_mag_log = torch.log(pred_mag + 1e-8)
        target_mag_log = torch.log(target_mag + 1e-8)
        
        # MSE
        freq_mse = F.mse_loss(pred_mag_log, target_mag_log)
        
        return freq_mse.item() if isinstance(freq_mse, torch.Tensor) else freq_mse
    
    @staticmethod
    def frequency_spectrum_psnr(pred, target):
        """
        频域幅度谱PSNR
        
        Args:
            pred: (H, W) or (B, C, H, W) - 预测图像
            target: (H, W) or (B, C, H, W) - 目标图像
        
        Returns:
            freq_psnr: float - 频域幅度谱PSNR (dB)
        """
        pred_t = ImageQualityMetrics._ensure_4d(pred)
        target_t = ImageQualityMetrics._ensure_4d(target)
        
        # FFT变换
        pred_fft = torch.fft.rfft2(pred_t, norm='ortho')
        target_fft = torch.fft.rfft2(target_t, norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 对数域
        pred_mag_log = torch.log(pred_mag + 1e-8)
        target_mag_log = torch.log(target_mag + 1e-8)
        
        # MSE
        mse = F.mse_loss(pred_mag_log, target_mag_log)
        
        if mse == 0:
            return float('inf')
        
        # PSNR
        max_val = target_mag_log.max() - target_mag_log.min()
        if max_val > 0:
            psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
            return psnr_value.item()
        else:
            return float('inf')
    
    @staticmethod
    def frequency_correlation(pred, target):
        """
        频域幅度谱相关系数
        
        Args:
            pred: (H, W) or (B, C, H, W) - 预测图像
            target: (H, W) or (B, C, H, W) - 目标图像
        
        Returns:
            freq_correlation: float - 频域相关系数 [-1, 1]，越接近1越好
        """
        pred_t = ImageQualityMetrics._ensure_4d(pred)
        target_t = ImageQualityMetrics._ensure_4d(target)
        
        # FFT变换
        pred_fft = torch.fft.rfft2(pred_t, norm='ortho')
        target_fft = torch.fft.rfft2(target_t, norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 展平
        pred_mag_flat = pred_mag.flatten(1)  # (B, H*W)
        target_mag_flat = target_mag.flatten(1)
        
        # 计算相关系数（对每个batch）
        correlations = []
        for i in range(pred_mag_flat.shape[0]):
            pred_vec = pred_mag_flat[i]
            target_vec = target_mag_flat[i]
            
            # 均值
            pred_mean = pred_vec.mean()
            target_mean = target_vec.mean()
            
            # 协方差
            numerator = ((pred_vec - pred_mean) * (target_vec - target_mean)).sum()
            
            # 标准差
            pred_std = torch.sqrt(((pred_vec - pred_mean) ** 2).sum() + 1e-8)
            target_std = torch.sqrt(((target_vec - target_mean) ** 2).sum() + 1e-8)
            
            # 相关系数
            corr = numerator / (pred_std * target_std + 1e-8)
            correlations.append(corr.item())
        
        return np.mean(correlations)
    
    @staticmethod
    def energy_distribution_similarity(pred, target, bins=256):
        """
        能量分布相似度（基于直方图的Bhattacharyya系数）
        
        Args:
            pred: (H, W) or (B, C, H, W) - 预测图像
            target: (H, W) or (B, C, H, W) - 目标图像
            bins: int - 直方图bin数量
        
        Returns:
            energy_sim: float - 能量分布相似度 [0, 1]，越接近1越好
        """
        # 转换为numpy
        pred_np = ImageQualityMetrics._to_numpy(pred)
        target_np = ImageQualityMetrics._to_numpy(target)
        
        # 展平
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        # 计算直方图
        pred_hist, _ = np.histogram(pred_flat, bins=bins, range=(0, 1))
        target_hist, _ = np.histogram(target_flat, bins=bins, range=(0, 1))
        
        # 归一化
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        # Bhattacharyya系数
        bc = np.sqrt(pred_hist * target_hist).sum()
        
        return float(bc)
    
    @staticmethod
    def compute_all_metrics(pred, target, include_frequency=True):
        """
        计算所有评估指标
        
        Args:
            pred: (H, W) or (B, C, H, W) - 预测图像
            target: (H, W) or (B, C, H, W) - 目标图像
            include_frequency: bool - 是否包含频域指标
        
        Returns:
            metrics: dict - 所有指标
        """
        metrics = {}
        
        # 基础指标
        metrics['mse'] = ImageQualityMetrics.mse(pred, target)
        metrics['mae'] = ImageQualityMetrics.mae(pred, target)
        metrics['psnr'] = ImageQualityMetrics.psnr_score(pred, target, data_range=1.0)
        metrics['ssim'] = ImageQualityMetrics.ssim_score(pred, target, data_range=1.0)
        
        # 频域指标（最重要）
        if include_frequency:
            metrics['freq_mse'] = ImageQualityMetrics.frequency_spectrum_mse(pred, target)
            metrics['freq_psnr'] = ImageQualityMetrics.frequency_spectrum_psnr(pred, target)
            metrics['freq_correlation'] = ImageQualityMetrics.frequency_correlation(pred, target)
        
        # 能量分布相似度
        metrics['energy_similarity'] = ImageQualityMetrics.energy_distribution_similarity(pred, target)
        
        return metrics


# 便捷函数
def compute_metrics(pred, target, include_frequency=True):
    """
    便捷函数：计算所有评估指标
    
    Args:
        pred: numpy array or torch.Tensor - 预测图像
        target: numpy array or torch.Tensor - 目标图像
        include_frequency: bool - 是否包含频域指标
    
    Returns:
        metrics: dict - 所有指标
    """
    return ImageQualityMetrics.compute_all_metrics(pred, target, include_frequency)


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("ImageQualityMetrics V3 测试")
    print("="*60)
    
    # 创建测试数据
    pred = np.random.rand(512, 512).astype(np.float32)
    target = pred + np.random.randn(512, 512).astype(np.float32) * 0.1
    target = np.clip(target, 0, 1)
    
    # 计算所有指标
    metrics = compute_metrics(pred, target, include_frequency=True)
    
    print("\n评估指标:")
    print("="*60)
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"\n频域指标:")
    print(f"  Freq MSE:        {metrics['freq_mse']:.6f}")
    print(f"  Freq PSNR:       {metrics['freq_psnr']:.2f} dB")
    print(f"  Freq Correlation: {metrics['freq_correlation']:.4f}")
    print(f"\n其他指标:")
    print(f"  Energy Similarity: {metrics['energy_similarity']:.4f}")
    print("="*60)

