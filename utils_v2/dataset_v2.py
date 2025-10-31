"""
Dataset V2 - 纯图像对数据集
只加载sim和real图像，不需要prompt
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RDPairDataset(Dataset):
    """
    RD图配对数据集 - 纯图像对
    
    目录结构:
        data_root/
        ├── sim/        # 仿真图
        │   ├── rd001.png
        │   ├── rd002.png
        │   └── ...
        └── real/       # 真实图
            ├── rd001.png
            ├── rd002.png
            └── ...
    
    特点:
    - 不再需要prompt/目录
    - 通过文件名匹配sim和real图像
    - 数据增强（可选）
    """
    def __init__(self, data_root, transform=None, augment=False):
        """
        Args:
            data_root: str - 数据根目录
            transform: torchvision.transforms - 图像变换
            augment: bool - 是否启用数据增强
        """
        self.data_root = Path(data_root)
        self.sim_dir = self.data_root / "sim"
        self.real_dir = self.data_root / "real"
        
        self.transform = transform
        self.augment = augment
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        print(f"[Dataset V2] 加载完成:")
        print(f"  数据根目录: {data_root}")
        print(f"  图像对数量: {len(self.data_list)}")
    
    def _load_data_list(self):
        """加载sim-real配对列表"""
        data_list = []
        
        # 检查目录是否存在
        if not self.sim_dir.exists():
            raise ValueError(f"仿真图目录不存在: {self.sim_dir}")
        if not self.real_dir.exists():
            raise ValueError(f"真实图目录不存在: {self.real_dir}")
        
        # 遍历sim目录
        sim_files = sorted(self.sim_dir.glob("*.png"))
        
        for sim_path in sim_files:
            # 根据文件名找对应的real图像
            real_path = self.real_dir / sim_path.name
            
            if not real_path.exists():
                print(f"[警告] 缺少对应的真实图: {real_path}")
                continue
            
            data_list.append({
                'sim_path': str(sim_path),
                'real_path': str(real_path),
                'name': sim_path.stem
            })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Returns:
            sim_image: (1, H, W) - 仿真图 tensor
            real_image: (1, H, W) - 真实图 tensor
            name: str - 文件名（用于可视化）
        """
        data = self.data_list[idx]
        
        # 加载图像
        sim_img = Image.open(data['sim_path']).convert('L')  # 灰度图
        real_img = Image.open(data['real_path']).convert('L')
        
        # Resize到512x512（统一尺寸，避免奇数尺寸问题）
        target_size = (512, 512)
        if sim_img.size != target_size:
            sim_img = sim_img.resize(target_size, Image.BILINEAR)
            real_img = real_img.resize(target_size, Image.BILINEAR)
        
        # 数据增强（可选）
        if self.augment:
            sim_img, real_img = self._augment(sim_img, real_img)
        
        # 转换为tensor
        if self.transform:
            sim_img = self.transform(sim_img)
            real_img = self.transform(real_img)
        else:
            # 默认变换: ToTensor + Normalize
            to_tensor = transforms.ToTensor()
            sim_img = to_tensor(sim_img)
            real_img = to_tensor(real_img)
        
        return sim_img, real_img, data['name']
    
    def _augment(self, sim_img, real_img):
        """
        数据增强（同时应用于sim和real）
        
        增强方式:
        - 水平翻转（概率50%）
        - 亮度调整（±10%）
        - 对比度调整（±10%）
        
        注意: 不做旋转和裁剪（RD图有方向性）
        """
        import random
        from PIL import ImageEnhance
        
        # 水平翻转
        if random.random() > 0.5:
            sim_img = sim_img.transpose(Image.FLIP_LEFT_RIGHT)
            real_img = real_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 亮度调整
        brightness_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Brightness(sim_img)
        sim_img = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Brightness(real_img)
        real_img = enhancer.enhance(brightness_factor)
        
        # 对比度调整
        contrast_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Contrast(sim_img)
        sim_img = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Contrast(real_img)
        real_img = enhancer.enhance(contrast_factor)
        
        return sim_img, real_img


def get_dataloader(data_root, batch_size, num_workers=4, shuffle=True, augment=False):
    """
    创建DataLoader
    
    Args:
        data_root: str - 数据根目录
        batch_size: int - 批大小
        num_workers: int - 加载进程数
        shuffle: bool - 是否打乱
        augment: bool - 是否数据增强
    
    Returns:
        dataloader: torch.utils.data.DataLoader
    """
    # 图像变换（标准化到[-1, 1]）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # [0,1] -> [-1,1]
    ])
    
    dataset = RDPairDataset(data_root, transform=transform, augment=augment)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if shuffle else False
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("RDPairDataset V2 测试")
    print("="*60)
    
    # 创建测试数据集
    data_root = "./dataset/train"
    
    try:
        dataloader = get_dataloader(
            data_root=data_root,
            batch_size=4,
            num_workers=0,
            shuffle=True,
            augment=False
        )
        
        print(f"\nDataLoader创建成功!")
        print(f"  总批次数: {len(dataloader)}")
        
        # 读取一个batch
        sim_batch, real_batch, names = next(iter(dataloader))
        
        print(f"\n第一个batch:")
        print(f"  sim_batch: {sim_batch.shape}, range=[{sim_batch.min():.2f}, {sim_batch.max():.2f}]")
        print(f"  real_batch: {real_batch.shape}, range=[{real_batch.min():.2f}, {real_batch.max():.2f}]")
        print(f"  names: {names}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print(f"请确保数据目录结构正确:")
        print(f"  {data_root}/sim/")
        print(f"  {data_root}/real/")
    
    print("\n" + "="*60)

