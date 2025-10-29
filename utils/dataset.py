"""
RD图数据加载器
加载sim-real配对的RD图数据和对应的prompt
"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class RDPairDataset(Dataset):
    """
    RD图配对数据集
    
    数据目录结构:
    data_root/
    ├── sim/           # 仿真RD图
    │   ├── rd001.png
    │   ├── rd002.png
    │   └── ...
    ├── real/          # 真实RD图
    │   ├── rd001.png
    │   ├── rd002.png
    │   └── ...
    └── prompt/        # 提示词文件（每个图片对应一个txt文件）
        ├── rd001.txt
        ├── rd002.txt
        └── ...
    
    每个prompt文件包含该图片的描述，例如：
    radar-RD-map; ... target number = 2, the first target: distance = 85m, velocity = 1.00m/s, ...
    """
    
    def __init__(
        self,
        data_root,
        img_size=512,
        split='train',
        transform=None
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.split = split
        
        # 数据路径
        self.sim_dir = self.data_root / 'sim'
        self.real_dir = self.data_root / 'real'
        self.prompt_dir = self.data_root / 'prompt'
        
        # 检查路径
        assert self.sim_dir.exists(), f"仿真图像目录不存在: {self.sim_dir}"
        assert self.real_dir.exists(), f"真实图像目录不存在: {self.real_dir}"
        assert self.prompt_dir.exists(), f"提示词目录不存在: {self.prompt_dir}"
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        print(f"✓ 数据集加载完成: {len(self.data_list)} 对样本")
        
        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # 归一化到[-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def _load_data_list(self):
        """加载配对的图像文件名列表（同时检查prompt文件）"""
        sim_files = sorted([f.stem for f in self.sim_dir.glob('*.png')])
        real_files = sorted([f.stem for f in self.real_dir.glob('*.png')])
        prompt_files = sorted([f.stem for f in self.prompt_dir.glob('*.txt')])
        
        # 找到三者都存在的文件
        sim_set = set(sim_files)
        real_set = set(real_files)
        prompt_set = set(prompt_files)
        
        # 完整配对的文件（sim + real + prompt）
        paired_files = list(sim_set & real_set & prompt_set)
        paired_files.sort()
        
        # 检查缺失情况
        missing_in_real = sim_set - real_set
        missing_in_sim = real_set - sim_set
        missing_in_prompt = (sim_set & real_set) - prompt_set
        
        if missing_in_real:
            print(f"⚠️  警告: {len(missing_in_real)} 个sim图像缺少对应的real图像")
        if missing_in_sim:
            print(f"⚠️  警告: {len(missing_in_sim)} 个real图像缺少对应的sim图像")
        if missing_in_prompt:
            print(f"⚠️  警告: {len(missing_in_prompt)} 个图像对缺少对应的prompt文件")
        
        if not paired_files:
            raise ValueError(f"未找到完整配对的数据！请检查 {self.data_root} 目录")
        
        return paired_files
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        返回一对数据
        
        Returns:
            dict: {
                'sim_rd': (3, H, W) 仿真RD图
                'real_rd': (3, H, W) 真实RD图
                'prompt': str 文本描述
                'filename': str 文件名
            }
        """
        filename = self.data_list[idx]
        
        # 加载图像
        sim_path = self.sim_dir / f"{filename}.png"
        real_path = self.real_dir / f"{filename}.png"
        prompt_path = self.prompt_dir / f"{filename}.txt"
        
        # 读取图像
        sim_img = Image.open(sim_path).convert('RGB')
        real_img = Image.open(real_path).convert('RGB')
        
        # 读取prompt
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        # 图像变换
        sim_rd = self.transform(sim_img)
        real_rd = self.transform(real_img)
        
        return {
            'sim_rd': sim_rd,
            'real_rd': real_rd,
            'prompt': prompt,
            'filename': filename
        }


def create_dataloader(
    data_root,
    batch_size=4,
    img_size=512,
    num_workers=4,
    shuffle=True,
    split='train'
):
    """
    便捷函数：创建DataLoader
    
    Args:
        data_root: 数据根目录
        batch_size: 批大小
        img_size: 图像尺寸
        num_workers: 工作进程数
        shuffle: 是否打乱
        split: 'train' 或 'val'
    
    Returns:
        dataloader: torch.utils.data.DataLoader
    """
    dataset = RDPairDataset(
        data_root=data_root,
        img_size=img_size,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == 'train' else False
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试（需要有数据才能运行）
    print("="*60)
    print("RD图数据加载器测试")
    print("="*60)
    print("\n数据目录结构应为：")
    print("data/")
    print("├── sim/       # 仿真RD图（PNG格式）")
    print("├── real/      # 真实RD图（PNG格式）")
    print("└── prompt/    # 提示词文件（每个图片对应一个TXT文件）")
    
    # 示例：创建测试数据
    test_data_root = Path("./test_data")
    if not test_data_root.exists():
        print(f"\n创建测试数据目录: {test_data_root}")
        (test_data_root / "sim").mkdir(parents=True, exist_ok=True)
        (test_data_root / "real").mkdir(parents=True, exist_ok=True)
        (test_data_root / "prompt").mkdir(parents=True, exist_ok=True)
        
        # 创建虚拟数据
        test_prompts = [
            "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 1, the first target: distance = 102m, velocity = 20.00m/s.",
            "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s.",
            "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 3, the first target: distance = 79m, velocity = -27.00m/s, the second target: distance = 126m, velocity = -18.00m/s, the third target: distance = 26m, velocity = 26.00m/s.",
        ]
        
        for i, prompt in enumerate(test_prompts):
            filename = f"rd{i+1:03d}"
            
            # 创建虚拟图像
            sim_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            real_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            sim_img.save(test_data_root / "sim" / f"{filename}.png")
            real_img.save(test_data_root / "real" / f"{filename}.png")
            
            # 创建prompt文件
            with open(test_data_root / "prompt" / f"{filename}.txt", "w", encoding='utf-8') as f:
                f.write(prompt)
        
        print(f"✓ 测试数据创建完成！共 {len(test_prompts)} 对样本")
    
    # 创建数据加载器
    try:
        print("\n测试数据加载...")
        dataloader = create_dataloader(
            data_root=test_data_root,
            batch_size=2,
            shuffle=False
        )
        
        # 加载一个batch
        batch = next(iter(dataloader))
        
        print(f"\n✓ 批次数据加载成功:")
        print(f"  sim_rd 形状: {batch['sim_rd'].shape}")
        print(f"  real_rd 形状: {batch['real_rd'].shape}")
        print(f"  文件名: {batch['filename']}")
        print(f"\n  Prompt示例 (第1个):")
        print(f"    {batch['prompt'][0][:100]}...")
        
        print("\n" + "="*60)
        print("✓ 数据加载器测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        print("请按照README.md准备数据后运行训练脚本")

