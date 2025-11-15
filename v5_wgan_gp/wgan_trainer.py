"""
V5 WGAN-GP训练器
基于V4训练器架构，集成WGAN-GP替代传统GAN
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models_v2.flow_matching_v2 import Sim2RealFlowModel
from utils_v2 import frequency_domain_loss, EarlyStopping
from critic_doppler import DopplerOnlyCritic, doppler_wgan_gp_loss, doppler_feature_matching_loss


class WGANTrainer:
    """
    V5 WGAN-GP训练器
    
    主要特点：
    1. 使用WGAN-GP替代传统GAN
    2. 更稳定的对抗训练
    3. 简化的超参数调节
    4. 保持多普勒专用特性
    """
    
    def __init__(self, config_path, pretrained_path):
        """
        初始化训练器
        
        Args:
            config_path: str - 配置文件路径
            pretrained_path: str - 预训练模型路径
        """
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设备
        self.device = torch.device(self.config['device'])
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.setup_directories()
        
        # 初始化模型
        self.setup_model()
        self.setup_critic()
        self.setup_optimizers()
        self.setup_training()
        
        # TensorBoard
        self.writer = SummaryWriter(self.config['paths']['log_dir'])
        
        print("✅ V5 WGAN-GP训练器初始化完成")
    
    def setup_directories(self):
        """创建输出目录"""
        for path_key in ['output_dir', 'log_dir', 'checkpoint_dir']:
            path = self.config['paths'][path_key]
            os.makedirs(path, exist_ok=True)
        print("✓ 输出目录创建完成")
    
    def setup_model(self):
        """加载预训练的Flow Matching模型"""
        model_cfg = self.config['model']
        
        self.model = Sim2RealFlowModel(
            base_channels=int(model_cfg['base_channels']),
            channel_mult=tuple(model_cfg['channel_mult']),
            time_embed_dim=int(model_cfg['time_embed_dim']),
            num_res_blocks=int(model_cfg['num_res_blocks']),
            attention_levels=tuple(model_cfg.get('attention_levels', [])),
            dropout=float(model_cfg.get('dropout', 0.0))
        ).to(self.device)
        
        # 加载预训练权重
        print(f"加载预训练模型: {self.pretrained_path}")
        checkpoint = torch.load(self.pretrained_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 设置参数冻结策略
        self.setup_parameter_groups()
        
        print("✓ Flow Matching模型加载完成")
    
    def setup_parameter_groups(self):
        """设置参数分组（与V4相同的策略）"""
        freeze_mode = self.config['finetune'].get('freeze_mode', 'selective')
        
        if freeze_mode == 'all_trainable':
            for param in self.model.parameters():
                param.requires_grad = True
            print("参数策略: 所有参数可训练")
        
        elif freeze_mode == 'freeze_encoder':
            for name, param in self.model.named_parameters():
                if 'sim_encoder' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("参数策略: 冻结编码器，其他可训练")
        
        elif freeze_mode == 'selective':
            for name, param in self.model.named_parameters():
                if 'sim_encoder' in name:
                    param.requires_grad = False
                elif 'time_embedding' in name or 'time_mlp' in name:
                    param.requires_grad = False
                elif 'down_blocks.0' in name or 'down_blocks.1' in name:
                    param.requires_grad = False
                elif 'up_blocks.3' in name or 'up_blocks.2' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = True
            print("参数策略: 选择性微调（冻结编码器+低频层）")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")
    
    def setup_critic(self):
        """创建WGAN-GP Critic"""
        critic_cfg = self.config['critic']
        
        self.critic = DopplerOnlyCritic(
            base_channels=int(critic_cfg.get('base_channels', 64)),
            dropout=float(critic_cfg.get('dropout', 0.3))
        ).to(self.device)
        
        print(f"✓ WGAN-GP Critic创建完成（base_channels={critic_cfg.get('base_channels', 64)}）")
    
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
        
        # Critic优化器（WGAN-GP推荐使用Adam，beta1=0, beta2=0.9）
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.config['finetune']['lr_critic']),
            betas=(0.0, 0.9)  # WGAN-GP推荐设置
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
        
        print(f"✓ 优化器创建完成")
        print(f"  生成器学习率: {self.config['finetune']['lr_generator']}")
        print(f"  Critic学习率: {self.config['finetune']['lr_critic']}")
    
    def setup_training(self):
        """设置训练参数"""
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # WGAN-GP训练参数
        self.critic_update_freq = int(self.config['finetune'].get('critic_update_freq', 5))
        self.lambda_gp = float(self.config['finetune'].get('lambda_gp', 10.0))
        self.wgan_weight = float(self.config['finetune'].get('wgan_weight', 1.0))
        self.feature_matching_weight = float(self.config['finetune'].get('feature_matching_weight', 1.0))
        self.frequency_weight = float(self.config['loss'].get('frequency_weight', 1.5))
        
        # 梯度累积参数
        self.gradient_accumulation_steps = int(self.config['train'].get('gradient_accumulation_steps', 1))
        
        print(f"\n✓ WGAN-GP训练配置:")
        print(f"  Batch Size: {self.config['train']['batch_size']}")
        print(f"  梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"  等效Batch Size: {self.config['train']['batch_size'] * self.gradient_accumulation_steps}")
        print(f"  Critic更新频率: {self.critic_update_freq}")
        print(f"  梯度惩罚系数: {self.lambda_gp}")
        print(f"  WGAN权重: {self.wgan_weight}")
        
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
        """训练一个epoch（WGAN-GP版本）"""
        self.model.train()
        self.critic.train()
        
        # 初始化梯度
        self.generator_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss_g = 0
        total_loss_c = 0
        total_loss_fm = 0
        total_loss_freq = 0
        total_loss_wgan = 0
        
        # Critic评分统计
        total_real_score = 0
        total_fake_score = 0
        total_score_gap = 0
        total_gp_loss = 0
        critic_updates = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (sim_images, real_images, _) in enumerate(pbar):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # ============================================================
            # 阶段1：训练Critic（每N步更新一次，支持梯度累积）
            # ============================================================
            if batch_idx % self.critic_update_freq == 0:
                # 生成假图像
                with torch.no_grad():
                    fake_images = self.model.generate(
                        sim_images,
                        ode_steps=int(self.config['finetune']['ode_steps']),
                        ode_method=self.config['finetune']['ode_method']
                    )
                
                # WGAN-GP Critic损失
                c_loss, c_info = doppler_wgan_gp_loss(
                    self.critic, real_images, fake_images, 
                    mode='critic', lambda_gp=self.lambda_gp
                )
                
                # 梯度累积：损失归一化
                c_loss = c_loss / self.gradient_accumulation_steps
                
                # 反向传播（梯度累加）
                c_loss.backward()
                
                # 达到累积步数，执行优化器更新
                if (batch_idx // self.critic_update_freq + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        float(self.config['train']['max_grad_norm'])
                    )
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                
                # 统计
                total_loss_c += c_loss.item() * self.gradient_accumulation_steps
                total_real_score += c_info['real_score']
                total_fake_score += c_info['fake_score']
                total_score_gap += c_info['score_gap']
                total_gp_loss += c_info['gp_loss']
                critic_updates += 1
                
                # TensorBoard记录
                if self.global_step % int(self.config['train']['log_interval']) == 0:
                    self.writer.add_scalar('train/loss_critic', c_loss.item() * self.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/real_score', c_info['real_score'], self.global_step)
                    self.writer.add_scalar('train/fake_score', c_info['fake_score'], self.global_step)
                    self.writer.add_scalar('train/score_gap', c_info['score_gap'], self.global_step)
                    self.writer.add_scalar('train/gradient_penalty', c_info['gp_loss'], self.global_step)
            
            # ============================================================
            # 阶段2：训练生成器（支持梯度累积）
            # ============================================================
            # Flow Matching Loss
            loss_fm = self.model.compute_loss(sim_images, real_images)
            
            # 获取预测
            predicted = self.model.generate(
                sim_images,
                ode_steps=int(self.config['finetune']['ode_steps']),
                ode_method=self.config['finetune']['ode_method']
            )
            
            # 频域Loss（保持原有能力）
            loss_freq = torch.tensor(0.0, device=self.device)
            if self.config['loss'].get('use_frequency', False):
                loss_freq = frequency_domain_loss(predicted, real_images)
            
            # WGAN-GP对抗损失
            loss_wgan, wgan_info = doppler_wgan_gp_loss(
                self.critic, real_images, predicted, mode='generator'
            )
            
            # 特征匹配损失（辅助）
            loss_fm_wgan, fm_info = doppler_feature_matching_loss(
                self.critic, real_images, predicted
            )
            
            # 总损失
            loss_g = (
                loss_fm +
                self.frequency_weight * loss_freq +
                self.wgan_weight * (
                    loss_wgan +
                    self.feature_matching_weight * loss_fm_wgan
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
            total_loss_wgan += loss_wgan.item()
            
            # TensorBoard日志
            if self.global_step % int(self.config['train']['log_interval']) == 0:
                self.writer.add_scalar('train/loss_generator', loss_g.item() * self.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar('train/loss_fm', loss_fm.item(), self.global_step)
                if loss_freq.item() > 0:
                    self.writer.add_scalar('train/loss_frequency', loss_freq.item(), self.global_step)
                self.writer.add_scalar('train/loss_wgan', loss_wgan.item(), self.global_step)
                self.writer.add_scalar('train/loss_feature_matching', loss_fm_wgan.item(), self.global_step)
            
            # 进度条
            postfix = {
                'G': f"{loss_g.item():.4f}",
                'FM': f"{loss_fm.item():.4f}",
                'WGAN': f"{loss_wgan.item():.4f}",
            }
            if batch_idx % self.critic_update_freq == 0 and critic_updates > 0:
                postfix['C'] = f"{c_loss.item():.4f}"
                postfix['Real'] = f"{c_info['real_score']:.3f}"
                postfix['Fake'] = f"{c_info['fake_score']:.3f}"
                postfix['Gap'] = f"{c_info['score_gap']:.3f}"
            pbar.set_postfix(postfix)
            
            self.global_step += 1
        
        # Epoch平均
        n_batches = len(train_loader)
        c_updates = n_batches // self.critic_update_freq
        
        # 损失需要乘以accumulation_steps恢复原始尺度
        avg_loss_g = (total_loss_g * self.gradient_accumulation_steps) / n_batches
        avg_loss_c = (total_loss_c * self.gradient_accumulation_steps) / c_updates if c_updates > 0 else 0
        
        # WGAN-GP统计
        avg_real_score = total_real_score / critic_updates if critic_updates > 0 else 0
        avg_fake_score = total_fake_score / critic_updates if critic_updates > 0 else 0
        avg_score_gap = total_score_gap / critic_updates if critic_updates > 0 else 0
        avg_gp_loss = total_gp_loss / critic_updates if critic_updates > 0 else 0
        
        print(f"\nEpoch {epoch} 总结:")
        print(f"  生成器损失: {avg_loss_g:.6f}")
        print(f"  Critic损失: {avg_loss_c:.6f}")
        print(f"  真实图像评分: {avg_real_score:.4f}")
        print(f"  生成图像评分: {avg_fake_score:.4f}")
        print(f"  评分差距: {avg_score_gap:.4f}")
        print(f"  梯度惩罚: {avg_gp_loss:.6f}")
        print(f"  Critic更新次数: {critic_updates}")
        
        return avg_loss_g
    
    def validate(self, epoch, val_loader):
        """验证"""
        self.model.eval()
        self.critic.eval()
        
        total_loss = 0
        total_loss_fm = 0
        total_loss_freq = 0
        total_real_score = 0
        total_fake_score = 0
        
        with torch.no_grad():
            for sim_images, real_images, _ in tqdm(val_loader, desc="验证"):
                sim_images = sim_images.to(self.device)
                real_images = real_images.to(self.device)
                
                # Flow Matching Loss
                loss_fm = self.model.compute_loss(sim_images, real_images)
                
                # 生成预测
                predicted = self.model.generate(
                    sim_images,
                    ode_steps=int(self.config['finetune']['ode_steps']),
                    ode_method=self.config['finetune']['ode_method']
                )
                
                # 频域Loss
                loss_freq = torch.tensor(0.0, device=self.device)
                if self.config['loss'].get('use_frequency', False):
                    loss_freq = frequency_domain_loss(predicted, real_images)
                
                # WGAN评分
                real_scores = self.critic(real_images)
                fake_scores = self.critic(predicted)
                
                # 总损失（验证时不包含对抗损失）
                loss = loss_fm + self.frequency_weight * loss_freq
                
                total_loss += loss.item()
                total_loss_fm += loss_fm.item()
                if loss_freq.item() > 0:
                    total_loss_freq += loss_freq.item()
                total_real_score += real_scores.mean().item()
                total_fake_score += fake_scores.mean().item()
        
        avg_loss = total_loss / len(val_loader)
        avg_real_score = total_real_score / len(val_loader)
        avg_fake_score = total_fake_score / len(val_loader)
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/loss_fm', total_loss_fm / len(val_loader), epoch)
        if total_loss_freq > 0:
            self.writer.add_scalar('val/loss_freq', total_loss_freq / len(val_loader), epoch)
        self.writer.add_scalar('val/real_score', avg_real_score, epoch)
        self.writer.add_scalar('val/fake_score', avg_fake_score, epoch)
        
        # 记录学习率
        current_lr = self.generator_optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        print(f"验证结果: Loss={avg_loss:.6f}, Real_Score={avg_real_score:.4f}, Fake_Score={avg_fake_score:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        if self.generator_scheduler:
            checkpoint['scheduler_state_dict'] = self.generator_scheduler.state_dict()
        
        # 保存常规检查点
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config['paths']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳模型: {best_path}")
        
        print(f"✅ 保存检查点: {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """主训练循环"""
        print(f"\n🚀 开始WGAN-GP训练...")
        print(f"训练轮数: {num_epochs}")
        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"验证集: {len(val_loader.dataset)} 样本")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            # 训练
            train_loss = self.train_one_epoch(epoch, train_loader)
            
            # 验证
            val_loss = self.validate(epoch, val_loader)
            
            # 学习率调度
            if self.generator_scheduler:
                self.generator_scheduler.step(val_loss)
                if epoch > 1:
                    current_lr = self.generator_optimizer.param_groups[0]['lr']
                    print(f"当前学习率: {current_lr:.2e}")
            
            # 检查是否最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"🎉 新的最佳验证损失: {val_loss:.6f}")
            
            # 保存检查点
            save_interval = int(self.config['train']['save_interval'])
            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if self.early_stopping:
                early_stop_triggered = self.early_stopping(val_loss)
                
                # 记录早停状态到TensorBoard
                self.writer.add_scalar('train/early_stopping_counter', self.early_stopping.counter, epoch)
                self.writer.add_scalar('train/early_stopping_patience', self.early_stopping.patience, epoch)
                if self.early_stopping.best_score is not None:
                    self.writer.add_scalar('train/early_stopping_best_score', self.early_stopping.best_score, epoch)
                
                if early_stop_triggered:
                    print(f"🛑 早停触发！在第 {epoch} 轮停止训练")
                    break
        
        # 保存最终模型
        final_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            'final_model.pth'
        )
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(final_checkpoint, final_path)
        
        print(f"\n✅ 训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"最终模型保存至: {final_path}")
        
        self.writer.close()
