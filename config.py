"""
配置加载器 - 从YAML文件加载配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置类 - 从config.yaml加载"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        加载配置文件
        
        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = config_path
        self._config = self._load_yaml(config_path)
        self._flatten_config()
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载YAML文件"""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _flatten_config(self):
        """将嵌套配置展平为类属性"""
        # 数据配置
        data_cfg = self._config.get('data', {})
        self.data_root = data_cfg.get('data_root', './data')
        self.val_data_root = data_cfg.get('val_data_root', None)
        self.img_size = data_cfg.get('img_size', 512)
        self.max_speed = data_cfg.get('max_speed', 20.0)
        self.max_range = data_cfg.get('max_range', 200.0)
        
        # 热力图配置
        heatmap_cfg = self._config.get('heatmap', {})
        self.heatmap_sigma = heatmap_cfg.get('sigma', 10.0)
        
        # Loss配置
        loss_cfg = self._config.get('loss', {})
        self.weight_factor = loss_cfg.get('weight_factor', 50)
        self.loss_threshold = loss_cfg.get('threshold', 0.1)
        self.loss_pool_size = loss_cfg.get('pool_size', 8)
        
        # 训练配置
        train_cfg = self._config.get('train', {})
        self.batch_size = train_cfg.get('batch_size', 4)
        self.num_epochs = train_cfg.get('num_epochs', 100)
        self.learning_rate = train_cfg.get('learning_rate', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 0.0)
        self.num_workers = train_cfg.get('num_workers', 4)
        self.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
        self.lr_scheduler = train_cfg.get('lr_scheduler', 'cosine')
        self.lr_warmup_epochs = train_cfg.get('lr_warmup_epochs', 5)
        self.noise_scale = train_cfg.get('noise_scale', 0.01)
        self.device = train_cfg.get('device', 'cuda')
        self.mixed_precision = train_cfg.get('mixed_precision', True)
        self.save_interval = train_cfg.get('save_interval', 5)
        self.log_interval = train_cfg.get('log_interval', 50)
        self.val_interval = train_cfg.get('val_interval', 1)
        self.save_best_only = train_cfg.get('save_best_only', False)
        self.keep_last_n_checkpoints = train_cfg.get('keep_last_n_checkpoints', 5)
        
        # 早停配置
        early_stop_cfg = train_cfg.get('early_stopping', {})
        self.early_stop_enabled = early_stop_cfg.get('enabled', True)
        self.early_stop_patience = early_stop_cfg.get('patience', 20)
        self.early_stop_min_delta = early_stop_cfg.get('min_delta', 0.0001)
        self.early_stop_monitor = early_stop_cfg.get('monitor', 'loss')
        
        # 推理配置
        inference_cfg = self._config.get('inference', {})
        self.ode_steps = inference_cfg.get('ode_steps', 20)
        self.ode_method = inference_cfg.get('ode_method', 'euler')
        
        # 路径配置
        paths_cfg = self._config.get('paths', {})
        self.output_dir = paths_cfg.get('output_dir', './outputs')
        self.log_dir = paths_cfg.get('log_dir', './logs')
        self.checkpoint_dir = paths_cfg.get('checkpoint_dir', './checkpoints')
        
        # 恢复训练
        resume_cfg = self._config.get('resume', {})
        self.resume = resume_cfg.get('checkpoint', None)
    
    def update(self, **kwargs):
        """更新配置（命令行参数覆盖）"""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
    
    def create_dirs(self):
        """创建必要的目录"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def display(self):
        """显示所有配置"""
        print("\n" + "="*60)
        print("配置信息")
        print("="*60)
        
        # 数据配置
        print("\n[数据配置]")
        print(f"  数据根目录: {self.data_root}")
        print(f"  图像尺寸: {self.img_size}")
        print(f"  最大速度: {self.max_speed} m/s")
        print(f"  最大距离: {self.max_range} m")
        
        # Loss配置
        print("\n[Loss配置]")
        print(f"  目标权重因子: {self.weight_factor}")
        print(f"  热力图阈值: {self.loss_threshold}")
        print(f"  Max Pool核大小: {self.loss_pool_size}")
        
        # 训练配置
        print("\n[训练配置]")
        print(f"  批大小: {self.batch_size}")
        print(f"  梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"  实际批大小: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"  训练轮数: {self.num_epochs}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  学习率调度: {self.lr_scheduler}")
        print(f"  设备: {self.device}")
        print(f"  混合精度: {self.mixed_precision}")
        print(f"  保存间隔: 每{self.save_interval}个epoch")
        print(f"  保留检查点数: {self.keep_last_n_checkpoints}")
        
        # 早停配置
        print("\n[早停机制]")
        print(f"  启用: {self.early_stop_enabled}")
        if self.early_stop_enabled:
            print(f"  监控指标: {self.early_stop_monitor}")
            print(f"  容忍轮数: {self.early_stop_patience}")
            print(f"  最小改善: {self.early_stop_min_delta}")
        
        # 推理配置
        print("\n[推理配置]")
        print(f"  ODE步数: {self.ode_steps}")
        print(f"  ODE方法: {self.ode_method}")
        
        # 路径配置
        print("\n[路径配置]")
        print(f"  检查点目录: {self.checkpoint_dir}")
        print(f"  日志目录: {self.log_dir}")
        
        print("="*60 + "\n")
    
    def save(self, save_path: str = None):
        """保存当前配置到YAML文件"""
        if save_path is None:
            save_path = "config_used.yaml"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"配置已保存到: {save_path}")


def load_config(config_path: str = "config.yaml") -> Config:
    """
    便捷函数：加载配置
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        Config对象
    """
    return Config(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = load_config("config.yaml")
    config.display()
    
    # 测试更新
    config.update(batch_size=8, learning_rate=5e-5)
    print("\n更新后:")
    print(f"  批大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
