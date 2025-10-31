"""
Flow Matching V2 模型插件示例

这是一个基于Flow Matching V2模型的推理插件实现
用户可以参考此插件编写自己的推理插件
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# 添加 backend 目录到 sys.path（用于导入 core）
backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from models_v2 import Sim2RealFlowModel

# 动态导入 InferenceInterface（兼容动态加载）
try:
    from core import InferenceInterface, InferenceMetrics
except ImportError:
    from webui.backend.core import InferenceInterface, InferenceMetrics


class FlowMatchingV2Plugin(InferenceInterface):
    """
    Flow Matching V2 推理插件
    
    配置参数:
        checkpoint_path: str - 模型检查点路径
        device: str - 设备 ('cuda:0', 'cuda:1', 'cpu')
        base_channels: int - 基础通道数 (默认64)
        channel_mult: tuple - 通道倍增 (默认(1,2,4,8))
        attention_levels: tuple - 注意力层级 (默认())
        image_size: tuple - 输入图像尺寸 (默认(512,512))
    """
    
    def __init__(self, plugin_name: str, config: Dict[str, Any]):
        super().__init__(plugin_name, config)
        
        # 推理参数
        self.base_channels = config.get('base_channels', 64)
        self.channel_mult = config.get('channel_mult', (1, 2, 4, 8))
        self.attention_levels = config.get('attention_levels', ())
        self.image_size = config.get('image_size', (512, 512))
        
        # 性能指标
        self.metrics = InferenceMetrics()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
    
    def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
        """加载模型"""
        try:
            # 创建模型
            self.model = Sim2RealFlowModel(
                base_channels=self.base_channels,
                channel_mult=self.channel_mult,
                attention_levels=self.attention_levels
            )
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # 移动到指定设备
            self.device = device
            self.model = self.model.to(device)
            self.model.eval()
            
            self.is_loaded = True
            
            print(f"✅ 模型加载成功: {checkpoint_path}")
            print(f"   设备: {device}")
            print(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.is_loaded = False
            return False
    
    def unload_model(self) -> bool:
        """卸载模型"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            print("✅ 模型卸载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型卸载失败: {e}")
            return False
    
    @torch.no_grad()
    def inference(
        self, 
        image_path: str,
        output_path: str,
        ode_steps: int = 50,
        ode_method: str = 'euler',
        **kwargs
    ) -> Dict[str, Any]:
        """单张图片推理"""
        if not self.is_loaded:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        try:
            start_time = time.time()
            
            # 加载图像
            image = Image.open(image_path).convert('L')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # ODE求解生成图像
            output_tensor = self._ode_solver(
                sim_image=image_tensor,
                ode_steps=ode_steps,
                method=ode_method
            )
            
            # 保存输出图像
            output_image = self._tensor_to_image(output_tensor[0])
            output_image.save(output_path)
            
            inference_time = time.time() - start_time
            self.metrics.update(inference_time)
            
            return {
                'success': True,
                'output_path': output_path,
                'inference_time': inference_time,
                'message': '推理成功',
                'metadata': {
                    'ode_steps': ode_steps,
                    'ode_method': ode_method,
                    'input_size': image_tensor.shape[-2:],
                    'output_size': output_tensor.shape[-2:]
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'推理失败: {str(e)}'
            }
    
    @torch.no_grad()
    def batch_inference(
        self,
        image_paths: List[str],
        output_dir: str,
        ode_steps: int = 50,
        ode_method: str = 'euler',
        **kwargs
    ) -> Dict[str, Any]:
        """批量推理"""
        if not self.is_loaded:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        succeeded = 0
        failed = 0
        start_time = time.time()
        
        for image_path in image_paths:
            # 生成输出文件名
            input_name = Path(image_path).stem
            output_path = output_dir / f"output_{input_name}.png"
            
            # 执行推理
            result = self.inference(
                image_path=image_path,
                output_path=str(output_path),
                ode_steps=ode_steps,
                ode_method=ode_method,
                **kwargs
            )
            
            results.append({
                'input_path': image_path,
                'output_path': str(output_path) if result['success'] else None,
                'success': result['success'],
                'inference_time': result.get('inference_time', 0),
                'message': result['message']
            })
            
            if result['success']:
                succeeded += 1
            else:
                failed += 1
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'total': len(image_paths),
            'succeeded': succeeded,
            'failed': failed,
            'results': results,
            'total_time': total_time,
            'avg_time': total_time / len(image_paths) if len(image_paths) > 0 else 0,
            'message': f'批量推理完成: {succeeded}/{len(image_paths)} 成功'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'name': 'Flow Matching V2',
            'version': '2.0',
            'description': 'Sim2Real雷达RD图转换模型（无需prompt）',
            'input_size': self.image_size,
            'output_size': self.image_size,
            'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp'],
            'default_params': {
                'ode_steps': 50,
                'ode_method': 'euler'
            },
            'custom_fields': {
                'base_channels': self.base_channels,
                'channel_mult': self.channel_mult,
                'attention_levels': self.attention_levels
            }
        }
        
        if self.is_loaded and self.model is not None:
            info['parameters'] = sum(p.numel() for p in self.model.parameters())
            info['performance'] = self.metrics.get_stats()
        
        return info
    
    def _ode_solver(
        self,
        sim_image: torch.Tensor,
        ode_steps: int = 50,
        method: str = 'euler'
    ) -> torch.Tensor:
        """ODE求解器（生成真实RD图）"""
        batch_size = sim_image.shape[0]
        
        # 初始化：从噪声开始
        x_t = torch.randn_like(sim_image)
        
        # 时间步长
        dt = 1.0 / ode_steps
        
        # 迭代求解
        for step in range(ode_steps):
            t = torch.ones(batch_size, device=self.device) * (step * dt)
            
            # 预测速度场
            v = self.model(x_t, t, sim_image)
            
            # 欧拉法更新
            if method == 'euler':
                x_t = x_t + v * dt
            # 可以添加其他ODE方法（如RK4）
            else:
                x_t = x_t + v * dt
        
        return x_t
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL Image"""
        # tensor: (C, H, W), 范围 [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if array.shape[0] == 1:
            # 灰度图
            array = array[0]
            return Image.fromarray(array, mode='L')
        else:
            # RGB图
            array = array.transpose(1, 2, 0)
            return Image.fromarray(array, mode='RGB')


# ============================================================================
# 插件测试代码（可选）
# ============================================================================
if __name__ == "__main__":
    # 测试插件
    config = {
        'checkpoint_path': str(project_root / 'outputs_v2/checkpoints/best_model.pth'),
        'device': 'cuda:0',
        'base_channels': 64,
        'channel_mult': (1, 2, 4, 8),
        'attention_levels': (),
        'image_size': (512, 512)
    }
    
    # 创建插件实例
    plugin = FlowMatchingV2Plugin('flow_matching_v2', config)
    
    # 加载模型
    if plugin.load_model(config['checkpoint_path']):
        print("\n模型信息:")
        import json
        print(json.dumps(plugin.get_model_info(), indent=2, ensure_ascii=False))
        
        # 测试推理
        test_image = project_root / 'dataset/sim/rd001.png'
        if test_image.exists():
            print(f"\n测试推理: {test_image}")
            result = plugin.inference(
                image_path=str(test_image),
                output_path='/tmp/test_output.png',
                ode_steps=50
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 卸载模型
        plugin.unload_model()

