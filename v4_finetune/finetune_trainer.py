"""
V4 - 两阶段微调训练器
阶段1：预训练Flow Matching基础模型（外部完成）
阶段2：加载预训练模型，用GAN专门优化多普勒效应
"""
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models_v2 import Sim2RealFlowModel
from utils_v2.losses import frequency_domain_loss, ssim_loss
from .discriminator_doppler import (
    DopplerOnlyDiscriminator,
    doppler_adversarial_loss,
    doppler_feature_matching_loss
)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.0001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class FineTuneTrainer:
    """
    两阶段微调训练器
    
    核心理念：
    1. 加载预训练的Flow Matching模型（基础能力已具备）
    2. 冻结大部分参数（保护背景和整体结构）
    3. 只微调与多普勒相关的高频特征
    4. 用判别器专门指导多普勒效应的改进
    """
    
    def __init__(self, config_path, pretrained_checkpoint):
        """
        Args:
            config_path: str - 配置文件路径
            pretrained_checkpoint: str - 预训练模型检查点路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['train']['device'])
        
        # 设置路径
        self.setup_paths()
        
        # 加载预训练模型
        self.load_pretrained_model(pretrained_checkpoint)
        
        # 创建判别器
        self.setup_discriminator()
        
        # 设置参数分组和优化器
        self.setup_optimizers()
        
        # 设置训练
        self.setup_training()
        
        print("="*60)
        print("V4 微调训练器初始化完成")
        print("="*60)
        print(f"预训练模型: {pretrained_checkpoint}")
        print(f"生成器总参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"生成器可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"判别器总参数: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"判别器可训练参数: {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad):,}")
        print("="*60)
    
    def setup_paths(self):
        """创建输出目录"""
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
    
    def load_pretrained_model(self, checkpoint_path):
        """加载预训练的Flow Matching模型"""
        print(f"\n加载预训练模型: {checkpoint_path}")
        
        # 创建模型
        model_cfg = self.config['model']
        self.model = Sim2RealFlowModel(
            base_channels=int(model_cfg['base_channels']),
            channel_mult=tuple(model_cfg['channel_mult']),
            time_embed_dim=int(model_cfg['time_embed_dim']),
            num_res_blocks=int(model_cfg['num_res_blocks']),
            attention_levels=tuple(model_cfg['attention_levels']),
            dropout=float(model_cfg['dropout'])
        ).to(self.device)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ 模型加载成功（Epoch {checkpoint['epoch']}）")
        
        # 参数分组策略
        self.setup_parameter_groups()
    
    def setup_parameter_groups(self):
        """
        设置参数分组
        
        策略：
        1. 编码器（SimEncoder）：冻结或极低学习率（背景特征已学好）
        2. UNet低频部分：低学习率（整体结构已稳定）
        3. UNet高频部分：正常学习率（多普勒相关，需要改进）
        4. Time Embedding：冻结（已学好）
        """
        freeze_mode = self.config['finetune'].get('freeze_mode', 'selective')
        
        if freeze_mode == 'all_trainable':
            # 模式1：所有参数都可训练（不推荐）
            for param in self.model.parameters():
                param.requires_grad = True
            print("参数策略: 所有参数可训练")
        
        elif freeze_mode == 'freeze_encoder':
            # 模式2：冻结编码器，其他可训练
            for name, param in self.model.named_parameters():
                if 'sim_encoder' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("参数策略: 冻结编码器，其他可训练")
        
        elif freeze_mode == 'selective':
            # 模式3：选择性微调（推荐）
            for name, param in self.model.named_parameters():
                if 'sim_encoder' in name:
                    # 编码器：冻结
                    param.requires_grad = False
                elif 'time_embedding' in name or 'time_mlp' in name:
                    # 时间嵌入：冻结
                    param.requires_grad = False
                elif 'down_blocks.0' in name or 'down_blocks.1' in name:
                    # UNet前两层下采样：冻结（低频特征）
                    param.requires_grad = False
                elif 'up_blocks.3' in name or 'up_blocks.2' in name:
                    # UNet后两层上采样：可训练（高频特征，多普勒相关）
                    param.requires_grad = True
                else:
                    # 其他：可训练
                    param.requires_grad = True
            print("参数策略: 选择性微调（冻结编码器+低频层）")
        
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")
    
    def setup_discriminator(self):
        """创建多普勒判别器"""
        disc_cfg = self.config['discriminator']
        
        self.discriminator = DopplerOnlyDiscriminator(
            base_channels=int(disc_cfg.get('base_channels', 64)),
            dropout=float(disc_cfg.get('dropout', 0.3))
        ).to(self.device)
        
        print(f"判别器已创建（base_channels={disc_cfg.get('base_channels', 64)}）")
    
    def setup_optimizers(self):
        """设置优化器"""
        # 生成器优化器（只优化可训练参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.generator_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.config['finetune']['lr_generator']),
            betas=tuple(self.config['train']['betas']),
            weight_decay=float(self.config['train']['weight_decay'])
        )
        
        # 判别器优化器
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.config['finetune']['lr_discriminator']),
            betas=(0.5, 0.999)  # GAN常用设置
        )
        
        # 学习率调度器
        lr_scheduler_cfg = self.config['finetune'].get('lr_scheduler', {})
        if lr_scheduler_cfg.get('enabled', True):
            self.generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.generator_optimizer,
                mode='min',
                factor=float(lr_scheduler_cfg.get('factor', 0.7)),
                patience=int(lr_scheduler_cfg.get('patience', 10)),
                min_lr=float(lr_scheduler_cfg.get('min_lr', 1e-7))
            )
        else:
            self.generator_scheduler = None
        
        print(f"优化器已创建")
        print(f"  生成器学习率: {self.config['finetune']['lr_generator']}")
        print(f"  判别器学习率: {self.config['finetune']['lr_discriminator']}")
        if self.generator_scheduler:
            print(f"  学习率调度器: ReduceLROnPlateau (factor={lr_scheduler_cfg.get('factor', 0.7)}, patience={lr_scheduler_cfg.get('patience', 10)})")
    
    def setup_training(self):
        """设置训练参数"""
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # GAN训练参数
        self.discriminator_update_freq = int(self.config['finetune'].get('discriminator_update_freq', 1))
        self.adversarial_weight = float(self.config['finetune'].get('adversarial_weight', 1.0))
        self.feature_matching_weight = float(self.config['finetune'].get('feature_matching_weight', 1.0))
        
        # 梯度累积参数
        self.gradient_accumulation_steps = int(self.config['train'].get('gradient_accumulation_steps', 1))
        
        print(f"\n训练配置:")
        print(f"  Batch Size: {self.config['train']['batch_size']}")
        print(f"  梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"  等效Batch Size: {self.config['train']['batch_size'] * self.gradient_accumulation_steps}")
        print(f"  判别器更新频率: {self.discriminator_update_freq}")
        
        # 早停机制
        early_stopping_cfg = self.config['finetune'].get('early_stopping', {})
        if early_stopping_cfg.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=int(early_stopping_cfg.get('patience', 20)),
                min_delta=float(early_stopping_cfg.get('min_delta', 0.0001)),
                monitor=early_stopping_cfg.get('monitor', 'val_loss')
            )
        else:
            self.early_stopping = None
    
    def train_one_epoch(self, epoch, train_loader):
        """训练一个epoch（支持梯度累积）"""
        self.model.train()
        self.discriminator.train()
        
        # 初始化梯度（梯度累积需要）
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        
        total_loss_g = 0
        total_loss_d = 0
        total_loss_fm = 0
        total_loss_freq = 0
        total_loss_adv = 0
        
        # 改进的准确率统计（累积样本数而不是batch数）
        total_correct_real = 0
        total_correct_fake = 0
        total_samples_discriminator = 0
        
        # 梯度累积计数器
        accum_steps = 0
        accum_correct_real = 0
        accum_correct_fake = 0
        accum_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (sim_images, real_images, _) in enumerate(pbar):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # ============================================================
            # 阶段1：训练判别器（每N步更新一次，支持梯度累积）
            # ============================================================
            if batch_idx % self.discriminator_update_freq == 0:
                # 生成假图像
                with torch.no_grad():
                    batch_size = sim_images.shape[0]
                    t_mid = torch.ones(batch_size, device=self.device) * 0.5
                    noise = torch.randn_like(real_images)
                    x_mid = 0.5 * noise + 0.5 * real_images
                    v_pred = self.model(x_mid, t_mid, sim_images)
                    fake_images = x_mid + v_pred * 0.5
                
                # 判别器损失
                d_loss, d_info = doppler_adversarial_loss(
                    self.discriminator, real_images, fake_images, mode='discriminator'
                )
                
                # 梯度累积：损失归一化
                d_loss = d_loss / self.gradient_accumulation_steps
                
                # 反向传播（梯度累加）
                d_loss.backward()
                
                # 累积统计信息
                accum_correct_real += d_info['num_correct_real']
                accum_correct_fake += d_info['num_correct_fake']
                accum_samples += d_info['num_samples']
                accum_steps += 1
                
                total_loss_d += d_loss.item() * self.gradient_accumulation_steps  # 恢复原始loss用于显示
                
                # 达到累积步数，执行优化器更新
                if accum_steps >= self.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        float(self.config['train']['max_grad_norm'])
                    )
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad()
                    
                    # 计算累积期内的准确率（多个样本的平均）
                    accum_real_acc = accum_correct_real / accum_samples if accum_samples > 0 else 0
                    accum_fake_acc = accum_correct_fake / accum_samples if accum_samples > 0 else 0
                    
                    # 记录累积期内的损失和准确率到TensorBoard（平滑值）
                    self.writer.add_scalar('train/loss_discriminator', d_loss.item() * self.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/d_real_acc', accum_real_acc, self.global_step)
                    self.writer.add_scalar('train/d_fake_acc', accum_fake_acc, self.global_step)
                    
                    # 累加到总统计
                    total_correct_real += accum_correct_real
                    total_correct_fake += accum_correct_fake
                    total_samples_discriminator += accum_samples
                    
                    # 重置累积器
                    accum_steps = 0
                    accum_correct_real = 0
                    accum_correct_fake = 0
                    accum_samples = 0
            
            # ============================================================
            # 阶段2：训练生成器（支持梯度累积）
            # ============================================================
            # Flow Matching Loss
            loss_fm = self.model.compute_loss(sim_images, real_images)
            
            # 获取预测
            batch_size = sim_images.shape[0]
            t_mid = torch.ones(batch_size, device=self.device) * 0.5
            noise = torch.randn_like(real_images)
            x_mid = 0.5 * noise + 0.5 * real_images
            v_pred = self.model(x_mid, t_mid, sim_images)
            predicted = x_mid + v_pred * 0.5
            
            # 频域Loss（保持原有能力）
            loss_freq = torch.tensor(0.0, device=self.device)
            if self.config['loss'].get('use_frequency', False):
                loss_freq = frequency_domain_loss(predicted, real_images)
            
            # GAN对抗损失（专门优化多普勒）
            loss_adv, adv_info = doppler_adversarial_loss(
                self.discriminator, real_images, predicted, mode='generator'
            )
            
            # GAN特征匹配损失（辅助）
            loss_fm_gan, fm_info = doppler_feature_matching_loss(
                self.discriminator, real_images, predicted
            )
            
            # 总损失
            loss_g = (
                loss_fm +
                float(self.config['loss'].get('frequency_weight', 2.0)) * loss_freq +
                float(self.config['finetune']['gan_weight']) * (
                    self.adversarial_weight * loss_adv +
                    self.feature_matching_weight * loss_fm_gan
                )
            )
            
            # 梯度累积：损失归一化
            loss_g = loss_g / self.gradient_accumulation_steps
            
            # 反向传播（梯度累加）
            loss_g.backward()
            
            # 每accumulation_steps步更新一次
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.config['train']['max_grad_norm'])
                )
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad()
            
            # 统计
            total_loss_g += loss_g.item()
            total_loss_fm += loss_fm.item()
            if loss_freq.item() > 0:
                total_loss_freq += loss_freq.item()
            total_loss_adv += loss_adv.item()
            
            # 日志
            if self.global_step % int(self.config['train']['log_interval']) == 0:
                self.writer.add_scalar('train/loss_generator', loss_g.item() * self.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar('train/loss_fm', loss_fm.item(), self.global_step)
                if loss_freq.item() > 0:
                    self.writer.add_scalar('train/loss_frequency', loss_freq.item(), self.global_step)
                self.writer.add_scalar('train/loss_adversarial', loss_adv.item(), self.global_step)
                self.writer.add_scalar('train/loss_feature_matching', loss_fm_gan.item(), self.global_step)
                # 判别器损失在累积期结束时记录，准确率也在那里记录
            
            # 进度条
            postfix = {
                'G': f"{loss_g.item():.4f}",
                'FM': f"{loss_fm.item():.4f}",
                'Adv': f"{loss_adv.item():.4f}",
            }
            if batch_idx % self.discriminator_update_freq == 0:
                postfix['D'] = f"{d_loss.item():.4f}"
                postfix['D_acc'] = f"{(d_info['real_acc'] + d_info['fake_acc'])/2:.2f}"
            pbar.set_postfix(postfix)
            
            self.global_step += 1
        
        # Epoch平均
        n_batches = len(train_loader)
        d_updates = n_batches // self.discriminator_update_freq
        
        # 损失需要乘以accumulation_steps恢复原始尺度
        avg_loss_g = (total_loss_g * self.gradient_accumulation_steps) / n_batches
        avg_loss_d = (total_loss_d * self.gradient_accumulation_steps) / d_updates if d_updates > 0 else 0
        
        # 使用改进的准确率计算（基于样本数而不是batch数）
        avg_real_acc = total_correct_real / total_samples_discriminator if total_samples_discriminator > 0 else 0
        avg_fake_acc = total_correct_fake / total_samples_discriminator if total_samples_discriminator > 0 else 0
        
        print(f"\nEpoch {epoch} 总结:")
        print(f"  生成器损失: {avg_loss_g:.6f}")
        print(f"  判别器损失: {avg_loss_d:.6f}")
        print(f"  判别器准确率: Real={avg_real_acc:.4f}, Fake={avg_fake_acc:.4f}")
        print(f"  统计样本数: {total_samples_discriminator}")
        
        return avg_loss_g
    
    @torch.no_grad()
    def validate(self, epoch, val_loader):
        """验证"""
        self.model.eval()
        self.discriminator.eval()
        
        total_loss = 0
        total_loss_fm = 0
        total_loss_freq = 0
        
        for sim_images, real_images, _ in tqdm(val_loader, desc="Validation"):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # Flow Matching Loss
            loss_fm = self.model.compute_loss(sim_images, real_images)
            
            # 获取预测
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
            
            # 总损失
            loss = loss_fm + float(self.config['loss'].get('frequency_weight', 2.0)) * loss_freq
            
            total_loss += loss.item()
            total_loss_fm += loss_fm.item()
            if loss_freq.item() > 0:
                total_loss_freq += loss_freq.item()
        
        avg_loss = total_loss / len(val_loader)
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/loss_fm', total_loss_fm / len(val_loader), epoch)
        if total_loss_freq > 0:
            self.writer.add_scalar('val/loss_freq', total_loss_freq / len(val_loader), epoch)
        
        # 记录学习率
        current_lr = self.generator_optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'generator_scheduler_state_dict': self.generator_scheduler.state_dict() if self.generator_scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / f"finetuned_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_finetuned.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型: {best_path}")
        
        # 清理旧检查点（只保留最近N个）
        keep_n = int(self.config['train'].get('keep_last_n_checkpoints', 5))
        if keep_n > 0:
            checkpoints = sorted(
                self.checkpoint_dir.glob("finetuned_epoch_*.pth"),
                key=lambda p: int(p.stem.split('_')[-1])
            )
            if len(checkpoints) > keep_n:
                for old_ckpt in checkpoints[:-keep_n]:
                    old_ckpt.unlink()
                    print(f"  清理旧检查点: {old_ckpt.name}")
    
    def save_final_model(self):
        """保存最终模型（使用最佳模型的权重）"""
        best_path = self.checkpoint_dir / "best_finetuned.pth"
        final_path = self.checkpoint_dir / "final_finetuned.pth"
        
        if best_path.exists():
            # 加载最佳模型
            best_checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            # 保存为最终模型
            torch.save(best_checkpoint, final_path)
            print(f"✓ 保存最终模型: {final_path}")
            print(f"  最终模型基于最佳epoch: {best_checkpoint['epoch']}")
        else:
            print("警告：最佳模型不存在，无法保存最终模型")
    
    def train(self, train_loader, val_loader, num_epochs):
        """主训练循环"""
        print("\n开始微调训练...")
        
        for epoch in range(self.start_epoch, num_epochs):
            # 训练
            train_loss = self.train_one_epoch(epoch, train_loader)
            
            # 验证
            val_loss = self.validate(epoch, val_loader)
            
            print(f"  Val Loss: {val_loss:.6f}")
            
            # 学习率调度
            if self.generator_scheduler:
                self.generator_scheduler.step(val_loss)
                current_lr = self.generator_optimizer.param_groups[0]['lr']
                if epoch > 0:
                    print(f"  当前学习率: {current_lr:.2e}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ 新的最佳模型！")
            
            if (epoch + 1) % int(self.config['train']['save_interval']) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if self.early_stopping:
                # 先调用早停检查（会更新counter）
                early_stop_triggered = self.early_stopping(val_loss)
                
                # 记录更新后的早停计数器到TensorBoard
                self.writer.add_scalar('train/early_stopping_counter', self.early_stopping.counter, epoch)
                self.writer.add_scalar('train/early_stopping_patience', self.early_stopping.patience, epoch)
                if self.early_stopping.best_score is not None:
                    self.writer.add_scalar('train/early_stopping_best_score', self.early_stopping.best_score, epoch)
                
                # 打印当前计数器状态
                if early_stop_triggered:
                    print(f"\n早停触发！最佳Val Loss: {self.best_val_loss:.6f}")
                    print(f"早停计数器达到: {self.early_stopping.counter}/{self.early_stopping.patience}")
                    # 早停时保存最终模型
                    self.save_final_model()
                    break
                else:
                    # 显示当前计数器（0表示验证损失有改善，>0表示连续N轮未改善）
                    print(f"  早停计数器: {self.early_stopping.counter}/{self.early_stopping.patience} (0=改善中, {self.early_stopping.patience}=触发)")
        else:
            # 正常训练完成，保存最终模型
            print("\n训练正常完成！")
            self.save_final_model()
        
        print(f"\n微调训练完成！")
        print(f"最佳Val Loss: {self.best_val_loss:.6f}")
        print(f"\n保存的模型:")
        print(f"  - best_finetuned.pth    (最佳模型)")
        print(f"  - final_finetuned.pth   (最终模型，用于测试)")
        
        self.writer.close()
