"""
通用Sim2Real推理接口

用户需要继承此类并实现所有抽象方法，即可将自己的模型接入WebUI
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class InferenceInterface(ABC):
    """
    Sim2Real推理接口基类
    
    用户需要实现：
    1. load_model() - 加载模型
    2. unload_model() - 卸载模型
    3. inference() - 单张图片推理
    4. batch_inference() - 批量推理
    5. get_model_info() - 获取模型信息
    """
    
    def __init__(self, plugin_name: str, config: Dict[str, Any]):
        """
        初始化推理接口
        
        Args:
            plugin_name: 插件名称（唯一标识）
            config: 配置字典，包含模型路径、设备等信息
        """
        self.plugin_name = plugin_name
        self.config = config
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
        """
        加载模型到指定设备
        
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备（'cuda:0', 'cuda:1', 'cpu'）
            
        Returns:
            bool: 加载成功返回True，否则False
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """
        卸载模型，释放显存
        
        Returns:
            bool: 卸载成功返回True，否则False
        """
        pass
    
    @abstractmethod
    def inference(
        self, 
        image_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        单张图片推理
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片保存路径
            **kwargs: 其他推理参数（如ode_steps, temperature等）
            
        Returns:
            Dict: 推理结果字典
            {
                'success': bool,
                'output_path': str,
                'inference_time': float,  # 推理耗时（秒）
                'message': str,           # 状态信息
                'metadata': Dict          # 其他元数据（可选）
            }
        """
        pass
    
    @abstractmethod
    def batch_inference(
        self,
        image_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量推理
        
        Args:
            image_paths: 输入图片路径列表
            output_dir: 输出目录
            **kwargs: 其他推理参数
            
        Returns:
            Dict: 批量推理结果
            {
                'success': bool,
                'total': int,
                'succeeded': int,
                'failed': int,
                'results': List[Dict],    # 每张图的推理结果
                'total_time': float,
                'message': str
            }
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
            {
                'name': str,
                'version': str,
                'description': str,
                'input_size': Tuple[int, int],      # (H, W)
                'output_size': Tuple[int, int],
                'parameters': int,                   # 参数量
                'supported_formats': List[str],      # 支持的图片格式
                'default_params': Dict,              # 默认推理参数
                'custom_fields': Dict                # 自定义字段
            }
        """
        pass
    
    def validate_config(self) -> bool:
        """
        验证配置是否合法（可选重写）
        
        Returns:
            bool: 配置合法返回True
        """
        required_keys = ['checkpoint_path', 'device']
        for key in required_keys:
            if key not in self.config:
                return False
        return True
    
    def preprocess(self, image: np.ndarray) -> Any:
        """
        预处理（可选重写）
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            预处理后的数据
        """
        return image
    
    def postprocess(self, output: Any) -> np.ndarray:
        """
        后处理（可选重写）
        
        Args:
            output: 模型输出
            
        Returns:
            后处理后的图像（numpy数组）
        """
        return output


class InferenceMetrics:
    """推理性能指标（可选使用）"""
    
    def __init__(self):
        self.total_inferences = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        
    def update(self, inference_time: float):
        """更新指标"""
        self.total_inferences += 1
        self.total_time += inference_time
        self.min_time = min(self.min_time, inference_time)
        self.max_time = max(self.max_time, inference_time)
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.total_inferences == 0:
            return {}
        
        return {
            'total_inferences': self.total_inferences,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.total_inferences,
            'min_time': self.min_time,
            'max_time': self.max_time
        }
    
    def reset(self):
        """重置指标"""
        self.__init__()

