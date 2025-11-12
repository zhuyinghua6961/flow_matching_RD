"""
训练脚本 V3 GAN版本 - 真正的对抗训练
关键改进：
1. 使用AdversarialDiscriminator替代原DopplerClutterDiscriminator
2. 实现真正的对抗训练（交替训练生成器和判别器）
3. 支持对抗损失 + 特征匹配损失
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
from utils_v3 import ImageQualityMetrics, ModelEvaluator
from utils_v3.discriminator_gan import (
    AdversarialDiscriminator,
    combined_gan_loss
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 导入V3的Trainer作为基类
import importlib
train_v3_module = importlib.import_module('train_v3')
EarlyStopping = train_v3_module.EarlyStopping


class TrainerV3GAN(train_v3_module.TrainerV3):
    """
    V3 GAN训练器 - 真正的对抗训练
    继承自TrainerV3，重写判别器和训练相关方法
    """
    
    def setup_discriminator(self):
        """设置判别器（重写）"""
        gan_cfg = self.config.get('gan', {})
        
        # 创建对抗判别器
        self.discriminator = AdversarialDiscriminator(
            doppler_weight=gan_cfg.get('doppler_weight', 0.75),
            clutter_weight=gan_cfg.get('clutter_weight', 0.25)
        ).to(self.device)
        
        # 判别器优化器（对抗训练必须启用）
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=gan_cfg.get('lr_discriminator', 2e-4),
            betas=(0.5, 0.999)  # GAN常用的beta值
        )
        
        # GAN训练参数
        self.gan_adversarial_weight = gan_cfg.get('adversarial_weight', 1.0)
        self.gan_feature_matching_weight = gan_cfg.get('feature_matching_weight', 1.0)
        self.discriminator_update_freq = gan_cfg.get('discriminator_update_freq', 1)  # 每N步更新判别器
        
        print(f"对抗判别器已创建")
        print(f"  多普勒权重: {gan_cfg.get('doppler_weight', 0.75)}")
        print(f"  地杂波权重: {gan_cfg.get('clutter_weight', 0.25)}")
        print(f"  对抗损失权重: {self.gan_adversarial_weight}")
        print(f"  特征匹配权重: {self.gan_feature_matching_weight}")
        print(f"  判别器更新频率: 每{self.discriminator_update_freq}步")
    
    def train_one_epoch(self, epoch):
        """训练一个epoch（重写以支持对抗训练）"""
        self.model.train()
        self.discriminator.train()  # 判别器也要训练
        
        total_loss = 0
        total_loss_fm = 0
        total_loss_frequency = 0
        total_loss_ssim = 0
        total_loss_gan = 0
        total_loss_d = 0  # 判别器损失
        total_d_real_acc = 0
        total_d_fake_acc = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['train']['num_epochs']}")
        
        for batch_idx, (sim_images, real_images, _) in enumerate(pbar):
            sim_images = sim_images.to(self.device)
            real_images = real_images.to(self.device)
            
            # ============================================================
            # 阶段1：训练判别器
            # ============================================================
            if batch_idx % self.discriminator_update_freq == 0:
                self.discriminator_optimizer.zero_grad()
                
                # 生成假图像（用于训练判别器）
                with torch.no_grad():
                    batch_size = sim_images.shape[0]
                    t_mid = torch.ones(batch_size, device=self.device) * 0.5
                    noise = torch.randn_like(real_images)
                    x_mid = 0.5 * noise + 0.5 * real_images
                    v_pred = self.model(x_mid, t_mid, sim_images)
                    fake_images = x_mid + v_pred * 0.5
                
                # 判别器损失（对抗损失）
                d_loss, d_info = combined_gan_loss(
                    self.discriminator,
                    real_images,
                    fake_images,
                    mode='discriminator',
                    adversarial_weight=self.gan_adversarial_weight,
                    feature_matching_weight=0.0,  # 判别器不需要特征匹配
                    doppler_weight=self.config.get('gan', {}).get('doppler_weight', 0.75),
                    clutter_weight=self.config.get('gan', {}).get('clutter_weight', 0.25)
                )
                
                # 反向传播并更新判别器
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config['train']['max_grad_norm']
                )
                self.discriminator_optimizer.step()
                
                # 记录判别器信息
                total_loss_d += d_loss.item()
                total_d_real_acc += d_info.get('adv_real_acc', 0)
                total_d_fake_acc += d_info.get('adv_fake_acc', 0)
            
            # ============================================================
            # 阶段2：训练生成器（Flow Matching模型）
            # ============================================================
            # 混合精度
            with torch.amp.autocast('cuda', enabled=self.config['train']['mixed_precision']):
                # Flow Matching Loss
                loss_fm = self.model.compute_loss(sim_images, real_images)
                
                # 获取预测样本
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
                
                # GAN Loss（对抗损失 + 特征匹配损失）
                loss_gan = torch.tensor(0.0, device=self.device)
                if self.use_gan and self.discriminator is not None:
                    g_loss, g_info = combined_gan_loss(
                        self.discriminator,
                        real_images,
                        predicted,
                        mode='generator',
                        adversarial_weight=self.gan_adversarial_weight,
                        feature_matching_weight=self.gan_feature_matching_weight,
                        doppler_weight=self.config.get('gan', {}).get('doppler_weight', 0.75),
                        clutter_weight=self.config.get('gan', {}).get('clutter_weight', 0.25)
                    )
                    loss_gan = g_loss
                
                # 总Loss（生成器）
                gan_weight = self.config.get('gan', {}).get('weights', {}).get('gan', 0.5)
                loss = (
                    loss_fm + 
                    self.config['loss'].get('frequency_weight', 2.0) * loss_freq +
                    self.config['loss'].get('ssim_weight', 0.5) * loss_ssim_val +
                    gan_weight * loss_gan
                )
                loss = loss / self.grad_accum_steps
            
            # 反向传播（生成器）
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
            if loss_freq.item() > 0:
                total_loss_frequency += loss_freq.item()
            if loss_ssim_val.item() > 0:
                total_loss_ssim += loss_ssim_val.item()
            if loss_gan.item() > 0:
                total_loss_gan += loss_gan.item()
            
            # 日志
            if self.global_step % self.config['train']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.grad_accum_steps, self.global_step)
                self.writer.add_scalar('train/loss_fm', loss_fm.item(), self.global_step)
                if loss_freq.item() > 0:
                    self.writer.add_scalar('train/loss_frequency', loss_freq.item(), self.global_step)
                if loss_ssim_val.item() > 0:
                    self.writer.add_scalar('train/loss_ssim', loss_ssim_val.item(), self.global_step)
                if loss_gan.item() > 0:
                    self.writer.add_scalar('train/loss_gan', loss_gan.item(), self.global_step)
                if batch_idx % self.discriminator_update_freq == 0:
                    self.writer.add_scalar('train/loss_discriminator', d_loss.item(), self.global_step)
                    self.writer.add_scalar('train/d_real_acc', d_info.get('adv_real_acc', 0), self.global_step)
                    self.writer.add_scalar('train/d_fake_acc', d_info.get('adv_fake_acc', 0), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # 进度条
            postfix_dict = {
                'G_loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'fm': f"{loss_fm.item():.4f}",
            }
            if loss_freq.item() > 0:
                postfix_dict['freq'] = f"{loss_freq.item():.4f}"
            if loss_gan.item() > 0:
                postfix_dict['gan'] = f"{loss_gan.item():.4f}"
            if batch_idx % self.discriminator_update_freq == 0:
                postfix_dict['D_loss'] = f"{d_loss.item():.4f}"
                postfix_dict['D_acc'] = f"{(d_info.get('adv_real_acc', 0) + d_info.get('adv_fake_acc', 0))/2:.2f}"
            postfix_dict['lr'] = f"{self.optimizer.param_groups[0]['lr']:.6f}"
            
            pbar.set_postfix(postfix_dict)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_fm = total_loss_fm / len(self.train_loader)
        
        # 打印epoch总结
        d_updates = len(self.train_loader) // self.discriminator_update_freq
        if d_updates > 0:
            print(f"\n  判别器平均损失: {total_loss_d / d_updates:.6f}")
            print(f"  判别器准确率: Real={total_d_real_acc / d_updates:.4f}, Fake={total_d_fake_acc / d_updates:.4f}")
        
        return avg_loss, avg_loss_fm


def main():
    parser = argparse.ArgumentParser(description="Train Sim2Real Flow Matching V3 with Adversarial GAN")
    parser.add_argument('--config', type=str, default='config_v2.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    print("="*60)
    print("使用真正的对抗训练（Adversarial GAN）")
    print("="*60)
    
    trainer = TrainerV3GAN(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
