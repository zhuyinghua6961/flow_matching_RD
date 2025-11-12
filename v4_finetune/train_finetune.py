"""
V4 微调训练脚本
使用预训练的Flow Matching模型，用GAN专门优化多普勒效应
"""
import sys
import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils_v2 import RDPairDataset
from v4_finetune.finetune_trainer import FineTuneTrainer


def main():
    parser = argparse.ArgumentParser(description="V4 Fine-tune Training with Doppler GAN")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练模型检查点路径（可选，优先级高于配置文件）')
    args = parser.parse_args()
    
    # 读取配置文件
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定预训练模型路径（命令行参数优先）
    pretrained_path = args.pretrained
    if pretrained_path is None:
        # 从配置文件读取
        pretrained_path = config.get('paths', {}).get('pretrained_model', None)
        if pretrained_path is None:
            raise ValueError("必须在配置文件中指定 paths.pretrained_model 或使用 --pretrained 参数")
        print(f"使用配置文件中的预训练模型: {pretrained_path}")
    else:
        print(f"使用命令行指定的预训练模型: {pretrained_path}")
    
    # 创建训练器
    trainer = FineTuneTrainer(args.config, pretrained_path)
    
    # 加载数据
    print("\n加载数据集...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[trainer.config['data']['normalize_mean']],
            std=[trainer.config['data']['normalize_std']]
        )
    ])
    
    # 训练集
    train_dataset = RDPairDataset(
        data_root=trainer.config['data']['train_root'],
        transform=transform,
        augment=trainer.config['data']['augment']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer.config['train']['batch_size'],
        shuffle=True,
        num_workers=trainer.config['train']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集
    val_dataset = RDPairDataset(
        data_root=trainer.config['data']['val_root'],
        transform=transform,
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer.config['train']['batch_size'],
        shuffle=False,
        num_workers=trainer.config['train']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 开始训练
    num_epochs = trainer.config['finetune']['num_epochs']
    trainer.train(train_loader, val_loader, num_epochs)


if __name__ == "__main__":
    main()
