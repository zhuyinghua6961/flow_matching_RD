"""
训练脚本 V2 - 纯图像对的Sim2Real Flow Matching
无需prompt，端到端训练
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models_v2 import Sim2RealFlowModel
from models_v2.perceptual_loss import PerceptualLoss
from utils_v2 import RDPairDataset
from utils_v2.losses import frequency_domain_loss, ssim_loss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class EarlyStopping:
    """早停机制
    
    对于loss指标（越小越好）：
    - 如果 score < best_score - min_delta：认为有改善，更新best_score并重置counter
    - 如果 score >= best_score - min_delta：认为没有改善或改善不足，counter增加
    - 当counter >= patience时，触发早停
    """
    def __init__(self, patience=20, min_delta=0.0001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Args:
            score: 当前指标值（对于loss，越小越好）
        Returns:
            bool: 是否触发早停
        """
        if self.best_score is None:
            # 第一个epoch，初始化best_score
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            # 没有改善或改善不足min_delta（对于loss：score越大表示越差）
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 有改善（score < best_score - min_delta，对于loss表示降低了至少min_delta）
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    def __init__(self, config_path):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_environment()
        self.setup_paths()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_training()
        
        print("="*60)
        print("训练配置:")
        print(f"  模型: Sim2RealFlowModel V2")
        print(f"  参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  训练集: {len(self.train_loader.dataset)}")
        print(f"  验证集: {len(self.val_loader.dataset)}")
        print(f"  批大小: {self.config['train']['batch_size']}")
        print(f"  梯度累积: {self.config['train']['gradient_accumulation_steps']}")
        print(f"  实际批大小: {self.config['train']['batch_size'] * self.config['train']['gradient_accumulation_steps']}")
        print(f"  学习率: {self.config['train']['learning_rate']}")
        print(f"  混合精度: {self.config['train']['mixed_precision']}")
        print(f"  感知损失: {self.config['loss']['use_perceptual']} (权重={self.config['loss']['perceptual_weight']})")
        print("="*60)
    
    def setup_environment(self):
        """设置环境"""
        # 随机种子
        seed = self.config.get('misc', {}).get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # CuDNN
        if self.config.get('misc', {}).get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        if self.config.get('misc', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
        
        # 设备
        self.device = torch.device(self.config['train']['device'])
        print(f"使用设备: {self.device}")
    
    def setup_paths(self):
        """创建输出目录"""
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
    
    def setup_model(self):
        """创建模型"""
        model_cfg = self.config['model']
        
        self.model = Sim2RealFlowModel(
            base_channels=model_cfg['base_channels'],
            channel_mult=tuple(model_cfg['channel_mult']),
            time_embed_dim=model_cfg['time_embed_dim'],
            num_res_blocks=model_cfg['num_res_blocks'],
            attention_levels=tuple(model_cfg['attention_levels']),
            dropout=model_cfg['dropout']
        ).to(self.device)
        
        # Perceptual Loss
        if self.config['loss']['use_perceptual']:
            self.perceptual_criterion = PerceptualLoss(
                feature_layers=tuple(self.config['loss']['perceptual_layers'])
            ).to(self.device)
        else:
            self.perceptual_criterion = None
        
        # 恢复训练
        self.start_epoch = 1  # 🔧 统一：epoch从1开始
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        if self.config['resume']['checkpoint']:
            self.load_checkpoint(self.config['resume']['checkpoint'])
    
    def setup_data(self):
        """创建数据加载器"""
        # 图像变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.config['data']['normalize_mean']],
                std=[self.config['data']['normalize_std']]
            )
        ])
        
        # 训练集
        train_dataset = RDPairDataset(
            data_root=self.config['data']['train_root'],
            transform=transform,
            augment=self.config['data']['augment']
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        # 验证集
        val_dataset = RDPairDataset(
            data_root=self.config['data']['val_root'],
            transform=transform,
            augment=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=self.config['train']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
    
    def setup_optimizer(self):
        """创建优化器和调度器"""
        # 优化器
        if self.config['train']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=tuple(self.config['train']['betas']),
                weight_decay=self.config['train']['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=tuple(self.config['train']['betas'])
            )
        
        # 学习率调度器
        if self.config['train']['lr_scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['num_epochs'] - self.config['train']['lr_warmup_epochs'],
                eta_min=self.config['train']['lr_min']
            )
        elif self.config['train']['lr_scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config['train']['lr_scheduler'] == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config['train'].get('plateau_mode', 'min'),
                factor=self.config['train'].get('plateau_factor', 0.5),
                patience=self.config['train'].get('plateau_patience', 10),
                min_lr=self.config['train'].get('plateau_min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if self.config['train']['mixed_precision'] else None
    
    def setup_training(self):
        """设置训练相关"""
        # 早停
        if self.config['train']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['train']['early_stopping']['patience'],
                min_delta=self.config['train']['early_stopping']['min_delta'],
                monitor=self.config['train']['early_stopping']['monitor']
            )
        else:
            self.early_stopping = None
        
        # 梯度累积
        self.grad_accum_steps = self.config['train']['gradient_accumulation_steps']
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_loss_fm = 0
        total_loss_perceptual = 0
        total_loss_frequency = 0
        total_loss_ssim = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['train']['num_epochs']}")
        
        for batch_idx, (sim_images, real_images, _) in enumerate(pbar):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # 混合精度
            with torch.amp.autocast('cuda', enabled=self.config['train']['mixed_precision']):
                # Flow Matching Loss
                loss_fm = self.model.compute_loss(sim_images, real_images)
                
                # 获取一个预测样本用于结构loss计算
                # 使用t=0.5时刻的预测作为中间结果（避免完整ODE求解）
                batch_size = sim_images.shape[0]
                t_mid = torch.ones(batch_size, device=self.device) * 0.5
                noise = torch.randn_like(real_images)
                x_mid = 0.5 * noise + 0.5 * real_images
                v_pred = self.model(x_mid, t_mid, sim_images)
                # 简单预测结果
                predicted = x_mid + v_pred * 0.5
                
                # Perceptual Loss（简化版）
                loss_perceptual = torch.tensor(0.0, device=self.device)
                if (self.perceptual_criterion is not None and 
                    self.global_step % self.config['loss']['perceptual_interval'] == 0):
                    loss_perceptual = self.perceptual_criterion(predicted, real_images)
                
                # 频域Loss（学习多普勒结构）
                loss_freq = torch.tensor(0.0, device=self.device)
                if self.config['loss'].get('use_frequency', False):
                    loss_freq = frequency_domain_loss(predicted, real_images)
                
                # SSIM Loss（结构相似性）
                loss_ssim_val = torch.tensor(0.0, device=self.device)
                if self.config['loss'].get('use_ssim', False):
                    loss_ssim_val = ssim_loss(predicted, real_images)
                
                # 总Loss
                loss = (
                    loss_fm + 
                    self.config['loss']['perceptual_weight'] * loss_perceptual +
                    self.config['loss'].get('frequency_weight', 0.1) * loss_freq +
                    self.config['loss'].get('ssim_weight', 0.3) * loss_ssim_val
                )
                loss = loss / self.grad_accum_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['train']['max_grad_norm']
                )
                
                # 更新参数
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.grad_accum_steps
            total_loss_fm += loss_fm.item()
            if loss_perceptual.item() > 0:
                total_loss_perceptual += loss_perceptual.item()
            if loss_freq.item() > 0:
                total_loss_frequency += loss_freq.item()
            if loss_ssim_val.item() > 0:
                total_loss_ssim += loss_ssim_val.item()
            
            # 日志
            if self.global_step % self.config['train']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar('train/loss_fm', loss_fm.item(), self.global_step)
                if loss_perceptual.item() > 0:
                    self.writer.add_scalar('train/loss_perceptual', loss_perceptual.item(), self.global_step)
                if loss_freq.item() > 0:
                    self.writer.add_scalar('train/loss_frequency', loss_freq.item(), self.global_step)
                if loss_ssim_val.item() > 0:
                    self.writer.add_scalar('train/loss_ssim', loss_ssim_val.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            postfix_dict = {
                'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'fm': f"{loss_fm.item():.4f}",
            }
            if loss_freq.item() > 0:
                postfix_dict['freq'] = f"{loss_freq.item():.4f}"
            if loss_ssim_val.item() > 0:
                postfix_dict['ssim'] = f"{loss_ssim_val.item():.4f}"
            postfix_dict['lr'] = f"{self.optimizer.param_groups[0]['lr']:.6f}"
            
            pbar.set_postfix(postfix_dict)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_fm = total_loss_fm / len(self.train_loader)
        
        return avg_loss, avg_loss_fm
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_loss_fm = 0
        total_loss_frequency = 0
        total_loss_ssim = 0
        
        for sim_images, real_images, _ in tqdm(self.val_loader, desc="Validation"):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # Flow Matching Loss
            loss_fm = self.model.compute_loss(sim_images, real_images)
            
            # 获取预测用于结构loss
            batch_size = sim_images.shape[0]
            t_mid = torch.ones(batch_size, device=self.device) * 0.5
            noise = torch.randn_like(real_images)
            x_mid = 0.5 * noise + 0.5 * real_images
            v_pred = self.model(x_mid, t_mid, sim_images)
            predicted = x_mid + v_pred * 0.5
            
            # 频域Loss
            loss_freq = torch.tensor(0.0, device=self.device)
            if self.config['loss'].get('use_frequency', False):
                loss_freq = frequency_domain_loss(predicted, real_images)
            
            # SSIM Loss
            loss_ssim_val = torch.tensor(0.0, device=self.device)
            if self.config['loss'].get('use_ssim', False):
                loss_ssim_val = ssim_loss(predicted, real_images)
            
            # 总loss（与训练时相同的权重）
            loss = (
                loss_fm + 
                self.config['loss'].get('frequency_weight', 0.1) * loss_freq +
                self.config['loss'].get('ssim_weight', 0.3) * loss_ssim_val
            )
            
            total_loss += loss.item()
            total_loss_fm += loss_fm.item()
            if loss_freq.item() > 0:
                total_loss_frequency += loss_freq.item()
            if loss_ssim_val.item() > 0:
                total_loss_ssim += loss_ssim_val.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_fm = total_loss_fm / len(self.val_loader)
        
        # 记录
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/loss_fm', avg_loss_fm, epoch)
        if total_loss_frequency > 0:
            self.writer.add_scalar('val/loss_frequency', total_loss_frequency / len(self.val_loader), epoch)
        if total_loss_ssim > 0:
            self.writer.add_scalar('val/loss_ssim', total_loss_ssim / len(self.val_loader), epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
        
        # 清理旧检查点
        keep_n = self.config['train']['keep_last_n_checkpoints']
        if keep_n > 0:
            # 按epoch数字排序，而不是字符串排序
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
                key=lambda p: int(p.stem.split('_')[-1])  # 提取epoch数字
            )
            # 删除旧的，保留最近N个
            for ckpt in checkpoints[:-keep_n]:
                ckpt.unlink()
                print(f"删除旧检查点: {ckpt.name}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def train(self):
        """主训练循环"""
        print("\n开始训练...")
        
        # 🔧 统一：确保训练完整的num_epochs个epoch（从1到num_epochs）
        for epoch in range(self.start_epoch, self.config['train']['num_epochs'] + 1):
            # 训练
            train_loss, train_loss_fm = self.train_one_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 记录epoch级别的指标
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.6f} (FM: {train_loss_fm:.6f})")
            print(f"  Val Loss: {val_loss:.6f}")
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                # 问题1修复：立即保存最佳模型，不依赖save_interval
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ 新的最佳模型！Val Loss: {self.best_val_loss:.6f}")
            
            # 定期保存检查点（非最佳模型时）
            if (epoch + 1) % self.config['train']['save_interval'] == 0:
                if not is_best:  # 如果不是最佳模型，才保存定期检查点
                    self.save_checkpoint(epoch, is_best=False)
            
            # 早停
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\n早停触发！最佳Val Loss: {self.best_val_loss:.6f}")
                    # 问题2修复：早停时确保最佳模型已保存，然后保存最终模型
                    best_model_path = self.checkpoint_dir / "best_model.pth"
                    if best_model_path.exists():
                        print(f"  最佳模型已保存在: {best_model_path}")
                        # 加载最佳模型的状态用于保存final_model
                        best_checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                        # 保存早停时的最终模型（使用最佳模型的状态）
                        final_path = self.checkpoint_dir / "final_model.pth"
                        final_checkpoint = {
                            'epoch': best_checkpoint['epoch'],  # 使用最佳模型的epoch
                            'global_step': best_checkpoint['global_step'],
                            'model_state_dict': best_checkpoint['model_state_dict'],  # 使用最佳模型的权重
                            'optimizer_state_dict': best_checkpoint['optimizer_state_dict'],
                            'scheduler_state_dict': best_checkpoint['scheduler_state_dict'],
                            'best_val_loss': self.best_val_loss,
                            'config': self.config,
                            'early_stopped': True
                        }
                        if self.scaler and 'scaler_state_dict' in best_checkpoint:
                            final_checkpoint['scaler_state_dict'] = best_checkpoint['scaler_state_dict']
                        torch.save(final_checkpoint, final_path)
                        print(f"  保存最终模型（早停，使用最佳模型权重）: {final_path}")
                    else:
                        # 如果最佳模型文件不存在（理论上不应该发生），保存当前模型
                        print(f"  警告：最佳模型文件不存在，保存当前模型作为最终模型")
                        final_path = self.checkpoint_dir / "final_model.pth"
                        checkpoint = {
                            'epoch': epoch,
                            'global_step': self.global_step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                            'best_val_loss': self.best_val_loss,
                            'config': self.config,
                            'early_stopped': True
                        }
                        if self.scaler:
                            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                        torch.save(checkpoint, final_path)
                        print(f"  保存最终模型（早停）: {final_path}")
                    break
        
        # 正常训练结束，保存最终模型
        else:
            # 问题2修复：正常训练结束时，也使用最佳模型作为最终模型
            final_path = self.checkpoint_dir / "final_model.pth"
            best_model_path = self.checkpoint_dir / "best_model.pth"
            if best_model_path.exists():
                print(f"\n训练完成，加载最佳模型作为最终模型...")
                best_checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                final_checkpoint = {
                    'epoch': best_checkpoint['epoch'],
                    'global_step': best_checkpoint['global_step'],
                    'model_state_dict': best_checkpoint['model_state_dict'],
                    'optimizer_state_dict': best_checkpoint['optimizer_state_dict'],
                    'scheduler_state_dict': best_checkpoint['scheduler_state_dict'],
                    'best_val_loss': self.best_val_loss,
                    'config': self.config,
                    'early_stopped': False
                }
                if self.scaler and 'scaler_state_dict' in best_checkpoint:
                    final_checkpoint['scaler_state_dict'] = best_checkpoint['scaler_state_dict']
                torch.save(final_checkpoint, final_path)
                print(f"保存最终模型（训练完成，使用最佳模型权重）: {final_path}")
            else:
                # 如果最佳模型文件不存在（理论上不应该发生），保存当前模型
                print(f"警告：最佳模型文件不存在，保存当前模型作为最终模型")
                checkpoint = {
                    'epoch': self.config['train']['num_epochs'] - 1,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': self.best_val_loss,
                    'config': self.config,
                    'early_stopped': False
                }
                if self.scaler:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                torch.save(checkpoint, final_path)
                print(f"保存最终模型（训练完成）: {final_path}")
        
        print("\n训练完成！")
        print(f"最佳Val Loss: {self.best_val_loss:.6f}")
        print(f"最佳模型: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"最终模型: {self.checkpoint_dir / 'final_model.pth'}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Sim2Real Flow Matching V2")
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()

