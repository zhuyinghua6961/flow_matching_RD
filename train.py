"""
训练脚本 - Flow Matching RD图Sim2Real模型
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models import FlowMatchingModel
from utils import (
    HeatmapGenerator,
    WeightedFlowMatchingLoss,
    create_dataloader,
    EarlyStopping
)
from config import load_config


class Trainer:
    """训练器"""
    
    def __init__(self, config):
        self.config = config
        config.create_dirs()
        
        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建模型
        # 固定的轻量级模型架构
        self.model = FlowMatchingModel(
            unet_base_channels=64,
            unet_channel_mult=(1, 2, 4, 8),
            controlnet_base_channels=64,  # 与UNet保持一致
            controlnet_channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_levels=(3,),  # 只在最低分辨率(64×64)使用attention，避免OOM
            dropout=0.0,
            time_emb_dim=256
        ).to(self.device)
        
        # 打印模型参数量
        params = self.model.get_num_parameters()
        print(f"\n模型参数量:")
        print(f"  UNet: {params['unet']:.2f}M")
        print(f"  ControlNet: {params['controlnet']:.2f}M")
        print(f"  Total: {params['total']:.2f}M\n")
        
        # 热力图生成器
        self.heatmap_generator = HeatmapGenerator(
            img_size=config.img_size,
            max_speed=config.max_speed,
            max_range=config.max_range,
            sigma=config.heatmap_sigma
        )
        
        # 损失函数
        self.criterion = WeightedFlowMatchingLoss(
            weight_factor=config.weight_factor,
            threshold=config.loss_threshold,
            pool_size=config.loss_pool_size,
            focal_gamma=config.focal_gamma,
            use_perceptual=config.use_perceptual,
            perceptual_weight=config.perceptual_weight
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # 数据加载器
        self.train_loader = create_dataloader(
            data_root=config.data_root,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers,
            shuffle=True,
            split='train'
        )
        
        # 验证集加载器（如果提供）
        self.val_loader = None
        if config.val_data_root:
            try:
                self.val_loader = create_dataloader(
                    data_root=config.val_data_root,
                    batch_size=config.batch_size,
                    img_size=config.img_size,
                    num_workers=config.num_workers,
                    shuffle=False,
                    split='val'
                )
                print(f"✓ 验证集已加载: {len(self.val_loader)} 批次")
            except Exception as e:
                print(f"⚠️  验证集加载失败: {e}")
                print(f"  将使用训练loss作为早停依据")
        else:
            print("⚠️  未配置验证集，将使用训练loss作为早停依据")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 早停机制
        self.early_stopping = None
        if config.early_stop_enabled:
            self.early_stopping = EarlyStopping(
                patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta,
                mode='min',
                verbose=True
            )
            print(f"早停机制已启用: 监控{config.early_stop_monitor}, "
                  f"容忍{config.early_stop_patience}个epoch")
        
        # 恢复训练
        if config.resume:
            self.load_checkpoint(config.resume)
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
        
        epoch_losses = {
            'loss': 0.0,
            'weighted_loss': 0.0,
            'base_loss': 0.0,
            'target_loss': 0.0,
            'bg_loss': 0.0
        }
        
        # 梯度累积
        accumulation_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()  # 在循环开始前清零一次
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            sim_rd = batch['sim_rd'].to(self.device)
            real_rd = batch['real_rd'].to(self.device)
            prompts = batch['prompt']
            
            # 生成热力图（训练时有一定概率不使用精确heatmap，增强泛化性）
            heatmaps = []
            for prompt in prompts:
                # 20%的概率使用弱化的或均匀的heatmap，让模型学会从图像本身判断
                if np.random.random() < 0.2:
                    # 使用均匀热力图，让模型不依赖heatmap
                    heatmap = torch.ones(1, self.config.img_size, self.config.img_size) * 0.3
                else:
                    heatmap = self.heatmap_generator(prompt)
                heatmaps.append(heatmap)
            heatmap = torch.stack(heatmaps).to(self.device)
            
            # Flow Matching前向过程
            batch_size = sim_rd.shape[0]
            
            # 采样时间步 t ~ Uniform(0, 1)
            t = torch.rand(batch_size, device=self.device)
            
            # 添加小噪声
            noise = torch.randn_like(sim_rd) * self.config.noise_scale
            
            # 线性插值路径: x_t = (1-t)*sim + t*real + noise
            t_expanded = t[:, None, None, None]
            x_t = (1 - t_expanded) * sim_rd + t_expanded * real_rd + noise
            
            # 速度场目标: v = real - sim
            v_target = real_rd - sim_rd
            
            # 混合精度训练
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # 前向传播
                    v_pred = self.model(x_t, t, sim_rd, heatmap)
                    
                    # 计算损失（除以累积步数）
                    loss_dict = self.criterion(v_pred, v_target, heatmap)
                    loss = loss_dict['loss'] / accumulation_steps
                
                # 反向传播（累积梯度）
                self.scaler.scale(loss).backward()
                
                # 梯度累积：每accumulation_steps步更新一次
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # 前向传播
                v_pred = self.model(x_t, t, sim_rd, heatmap)
                
                # 计算损失（除以累积步数）
                loss_dict = self.criterion(v_pred, v_target, heatmap)
                loss = loss_dict['loss'] / accumulation_steps
                
                # 反向传播（累积梯度）
                loss.backward()
                
                # 梯度累积：每accumulation_steps步更新一次
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 累积损失（记录原始loss，不是缩放后的）
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'target': loss_dict['target_loss'].item(),
                'bg': loss_dict['bg_loss'].item(),
                'accum': f"{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}"
            })
            
            # 记录到TensorBoard
            if batch_idx % self.config.log_interval == 0:
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 处理最后不足accumulation_steps的batch
        if len(self.train_loader) % accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate_one_epoch(self, epoch):
        """在验证集上评估一个epoch"""
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            return None
        
        self.model.eval()
        
        val_losses = {
            'loss': 0.0,
            'weighted_loss': 0.0,
            'base_loss': 0.0,
            'target_loss': 0.0,
            'bg_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                # 数据移到设备
                sim_rd = batch['sim_rd'].to(self.device)
                real_rd = batch['real_rd'].to(self.device)
                prompts = batch['prompt']
                
                # 生成热力图
                heatmaps = []
                for prompt in prompts:
                    heatmap = self.heatmap_generator(prompt)
                    heatmaps.append(heatmap)
                heatmap = torch.stack(heatmaps).to(self.device)
                
                # Flow Matching前向过程
                batch_size = sim_rd.shape[0]
                t = torch.rand(batch_size, device=self.device)
                
                # 构造 x_t = (1-t)*x_0 + t*x_1 + noise
                t_expanded = t.view(-1, 1, 1, 1)
                noise = torch.randn_like(sim_rd) * self.config.noise_scale
                x_t = (1 - t_expanded) * sim_rd + t_expanded * real_rd + noise
                
                # 速度场目标
                v_target = real_rd - sim_rd
                
                # 前向传播
                v_pred = self.model(x_t, t, sim_rd, heatmap)
                
                # 计算损失
                loss_dict = self.criterion(v_pred, v_target, heatmap)
                
                # 累积损失
                for key in val_losses.keys():
                    if key in loss_dict:
                        val_losses[key] += loss_dict[key].item()
        
        # 平均损失
        for key in val_losses.keys():
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def save_checkpoint(self, epoch, loss=None, is_best=False, filename=None):
        """
        保存检查点（按epoch保存）
        
        Args:
            epoch: 当前epoch
            loss: 当前损失（用于判断是否为最佳）
            is_best: 是否为最佳模型
            filename: 指定文件名（可选）
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        
        save_path = Path(self.config.checkpoint_dir) / filename
        
        # 构建检查点
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存早停状态
        if self.early_stopping:
            checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
        
        # 保存检查点
        torch.save(checkpoint, save_path)
        print(f"✓ 检查点已保存: {save_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"⭐ 最佳模型已保存: {best_path} (loss: {loss:.6f})")
        
        # 清理旧检查点（保留最近N个）
        if self.config.keep_last_n_checkpoints > 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近N个"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # 获取所有epoch检查点（排除latest.pth和best_model.pth）
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pth")],
            key=lambda x: x.stat().st_mtime
        )
        
        # 删除多余的检查点
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            to_delete = checkpoints[:-self.config.keep_last_n_checkpoints]
            for ckpt in to_delete:
                ckpt.unlink()
                print(f"  已删除旧检查点: {ckpt.name}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点并恢复训练状态"""
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型和优化器
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载混合精度scaler
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复训练状态
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # 恢复早停状态
        if self.early_stopping and 'early_stopping_state' in checkpoint:
            self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            print(f"早停状态已恢复: 最佳loss={self.early_stopping.best_score:.6f}, "
                  f"等待计数={self.early_stopping.counter}")
        
        print(f"✓ 训练已恢复")
        print(f"  开始epoch: {self.start_epoch}")
        print(f"  全局步数: {self.global_step}")
        print(f"  最佳loss: {self.best_loss:.6f}")
    
    def train(self):
        """完整训练流程"""
        print("\n开始训练...")
        self.config.display()
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            # 训练一个epoch
            epoch_losses = self.train_one_epoch(epoch)
            
            # 验证一个epoch（如果有验证集）
            val_losses = self.validate_one_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 选择用于早停和保存模型的loss（优先使用验证loss）
            if val_losses is not None:
                current_loss = val_losses['loss']
                monitor_metric = val_losses.get(
                    self.config.early_stop_monitor, 
                    current_loss
                )
            else:
                current_loss = epoch_losses['loss']
                monitor_metric = epoch_losses.get(
                    self.config.early_stop_monitor, 
                    current_loss
                )
            
            # 判断是否为最佳模型
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # 打印epoch统计
            print(f"\nEpoch {epoch}/{self.config.num_epochs} 完成:")
            print(f"  [Train] Loss: {epoch_losses['loss']:.6f}, " + 
                  f"Target: {epoch_losses['target_loss']:.6f}, " +
                  f"BG: {epoch_losses['bg_loss']:.6f}")
            if val_losses is not None:
                print(f"  [Val]   Loss: {val_losses['loss']:.6f} " + 
                      (f"⭐ (最佳)" if is_best else ""))
                print(f"          Target: {val_losses['target_loss']:.6f}, " +
                      f"BG: {val_losses['bg_loss']:.6f}")
            else:
                print(f"  " + (f"⭐ (最佳)" if is_best else ""))
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 按epoch保存检查点
            should_save = (epoch + 1) % self.config.save_interval == 0
            
            if self.config.save_best_only:
                # 只保存最佳模型
                if is_best:
                    self.save_checkpoint(epoch, current_loss, is_best=True)
            else:
                # 定期保存 + 保存最佳
                if should_save:
                    self.save_checkpoint(epoch, current_loss, is_best=is_best)
                elif is_best:
                    self.save_checkpoint(epoch, current_loss, is_best=True)
            
            # 始终保存最新模型
            self.save_checkpoint(epoch, current_loss, is_best=False, filename='latest.pth')
            
            # 记录到TensorBoard
            for key, value in epoch_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            if val_losses is not None:
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            self.writer.add_scalar('epoch/best_loss', self.best_loss, epoch)
            
            # 早停检查
            if self.early_stopping:
                should_stop = self.early_stopping(monitor_metric, epoch)
                if should_stop:
                    print(f"\n⚠️  训练提前停止于epoch {epoch}")
                    print(f"  原因: {self.config.early_stop_monitor}在"
                          f"{self.config.early_stop_patience}个epoch内无改善")
                    print(f"  最佳epoch: {self.early_stopping.best_epoch}")
                    print(f"  最佳{self.config.early_stop_monitor}: "
                          f"{self.early_stopping.best_score:.6f}")
                    
                    # 保存最终检查点
                    self.save_checkpoint(epoch, current_loss, filename='early_stopped.pth')
                    break
        
        print("\n训练完成！")
        print(f"最佳loss: {self.best_loss:.6f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="训练Flow Matching RD图Sim2Real模型")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--data_root', type=str, default=None, help='数据根目录（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小（覆盖配置文件）')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（覆盖配置文件）')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    config.update(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        resume=args.resume,
        device=args.device
    )
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

