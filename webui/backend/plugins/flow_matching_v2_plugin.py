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
        
        # 设备配置（在load_model之前初始化）
        self.device = config.get('device', 'cuda:0')
        
        # 推理参数
        self.base_channels = config.get('base_channels', 64)
        self.channel_mult = config.get('channel_mult', (1, 2, 4, 8))
        self.attention_levels = config.get('attention_levels', ())
        self.image_size = config.get('image_size', (512, 512))
        
        # 可视化参数
        self.apply_colormap = config.get('apply_colormap', True)  # 是否应用伪彩色
        self.colormap_name = config.get('colormap_name', 'jet')   # colormap类型: jet, viridis, hot, etc.
        
        # 性能指标
        self.metrics = InferenceMetrics()
        
        # 图像预处理（与训练时保持一致）
        # ⚠️ 重要：这些参数必须和训练时config_v2.yaml中的normalize_mean/std一致！
        self.normalize_mean = config.get('normalize_mean', 0.35)  # 新数据集的均值
        self.normalize_std = config.get('normalize_std', 0.06)    # 新数据集的标准差
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.normalize_mean], std=[self.normalize_std])
        ])
    
    def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
        """加载模型"""
        try:
            print(f"[DEBUG] 开始加载模型: {checkpoint_path}")
            
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
            print(f"[DEBUG] 模型权重加载成功")
            
            # 移动到指定设备
            self.device = device
            self.model = self.model.to(device)
            self.model.eval()
            self.is_loaded = True
            
            # 输出模型的一些关键参数作为验证
            first_param = next(iter(self.model.parameters()))
            param_sum = first_param.sum().item()
            
            print(f"✅ 模型加载成功: {self.plugin_name}")
            print(f"   参数校验和: {param_sum:.6f}, 参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            import traceback
            print(f"❌ 模型加载失败: {e}")
            print(f"[DEBUG] 详细错误:")
            traceback.print_exc()
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
        apply_colormap: bool = None,
        colormap: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        单张图片推理
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            ode_steps: ODE求解步数
            ode_method: ODE求解方法
            apply_colormap: 是否应用伪彩色（None=使用配置）
            colormap: colormap名称（如'jet', 'viridis'等，None=使用配置）
        """
        print(f"[DEBUG] inference 被调用")
        print(f"[DEBUG] 插件名称: {self.plugin_name}")
        print(f"[DEBUG] 检查点路径: {self.config.get('checkpoint_path', 'Unknown')}")
        print(f"[DEBUG] is_loaded: {self.is_loaded}")
        print(f"[DEBUG] device: {self.device}")
        
        # 输出模型的一些关键参数来验证是否是不同的模型
        if self.model is not None:
            try:
                # 获取第一层的权重作为模型标识
                first_param = next(iter(self.model.parameters()))
                param_sum = first_param.sum().item()
                print(f"[DEBUG] 模型参数校验和: {param_sum:.6f}")
                print(f"[DEBUG] 模型参数形状: {first_param.shape}")
            except:
                print(f"[DEBUG] 无法获取模型参数信息")
        
        if not self.is_loaded:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        try:
            start_time = time.time()
            
            print(f"[DEBUG] 开始推理: {image_path}")
            
            # 临时修改colormap配置
            old_colormap = self.colormap_name
            if colormap is not None:
                self.colormap_name = colormap
            
            # 加载并预处理图像
            image = Image.open(image_path).convert('L')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # ODE求解生成图像
            output_tensor = self._ode_solver(
                sim_image=image_tensor,
                ode_steps=ode_steps,
                method=ode_method
            )
            
            # 添加输出统计信息用于验证
            output_mean = output_tensor.mean().item()
            output_std = output_tensor.std().item()
            print(f"[DEBUG] 输出统计: 均值={output_mean:.6f}, 标准差={output_std:.6f}")
            
            # 保存输出图像
            output_image = self._tensor_to_image(output_tensor[0], apply_colormap=apply_colormap)
            output_image.save(output_path)
            
            # 恢复colormap配置
            if colormap is not None:
                self.colormap_name = old_colormap
            
            inference_time = time.time() - start_time
            self.metrics.update(inference_time)
            
            print(f"[DEBUG] 推理成功，耗时: {inference_time:.2f}s")
            
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
            import traceback
            print(f"[DEBUG] 推理异常: {e}")
            print(f"[DEBUG] 详细错误:")
            traceback.print_exc()
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
            'description': 'Sim2Real雷达RD图转换模型（无需prompt，支持伪彩色可视化）',
            'input_size': self.image_size,
            'output_size': self.image_size,
            'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp'],
            'default_params': {
                'ode_steps': 50,
                'ode_method': 'euler',
                'apply_colormap': self.apply_colormap,
                'colormap': self.colormap_name
            },
            'custom_fields': {
                'base_channels': self.base_channels,
                'channel_mult': self.channel_mult,
                'attention_levels': self.attention_levels
            },
            'colormap_options': {
                'enabled': self.apply_colormap,
                'current': self.colormap_name,
                'available': ['jet', 'viridis', 'plasma', 'inferno', 'magma', 
                             'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
                             'gray', 'bone', 'copper', 'turbo']
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
    
    def _tensor_to_image(self, tensor: torch.Tensor, apply_colormap: bool = None) -> Image.Image:
        """
        将tensor转换为PIL Image
        
        Args:
            tensor: (C, H, W), 标准化后的值（模型输出）
            apply_colormap: 是否应用伪彩色映射（None=使用配置，用于RD图可视化）
        """
        # Denormalize: 标准化空间 -> [0, 1]
        # 反向操作: x_normalized = (x - mean) / std  =>  x = x_normalized * std + mean
        tensor = tensor * self.normalize_std + self.normalize_mean
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if array.shape[0] == 1:
            # 灰度图
            array = array[0]
            
            # 确定是否应用colormap
            if apply_colormap is None:
                apply_colormap = self.apply_colormap
            
            # 应用伪彩色映射（RD图可视化）
            if apply_colormap:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                
                # 使用配置的colormap
                try:
                    colormap = cm.get_cmap(self.colormap_name)
                except:
                    # 如果colormap不存在，使用默认的jet
                    colormap = cm.get_cmap('jet')
                    print(f"⚠️  Colormap '{self.colormap_name}' 不存在，使用 'jet'")
                
                # 归一化到[0, 1]
                normalized = array / 255.0
                # 应用colormap
                colored = colormap(normalized)
                # 转换为RGB (去掉alpha通道)
                colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
                return Image.fromarray(colored_rgb, mode='RGB')
            else:
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

