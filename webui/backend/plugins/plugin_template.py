"""
插件模板

用户可以复制此文件作为开发新插件的起点

使用步骤:
1. 复制此文件并重命名（如 my_model_plugin.py）
2. 修改类名（如 MyModelPlugin）
3. 实现所有必需的抽象方法
4. 在 WebUI 中注册插件
"""
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

# 导入你的模型
# import torch
# from your_model import YourModel

import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# 添加 backend 目录到 sys.path（用于导入 core）
backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# 动态导入 InferenceInterface（兼容动态加载）
try:
    from core import InferenceInterface, InferenceMetrics
except ImportError:
    from webui.backend.core import InferenceInterface, InferenceMetrics


class PluginTemplate(InferenceInterface):
    """
    插件模板类
    
    必需配置参数:
        checkpoint_path: str - 模型检查点路径
        device: str - 设备 ('cuda:0', 'cuda:1', 'cpu')
    
    可选配置参数:
        （根据你的模型添加）
    """
    
    def __init__(self, plugin_name: str, config: Dict[str, Any]):
        super().__init__(plugin_name, config)
        
        # 初始化你的参数
        self.your_param = config.get('your_param', default_value)
        
        # 性能指标（可选）
        self.metrics = InferenceMetrics()
    
    def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
        """
        加载模型
        
        必须实现：
        1. 创建模型实例
        2. 加载权重
        3. 移动到指定设备
        4. 设置为评估模式
        5. 设置 self.is_loaded = True
        """
        try:
            # TODO: 实现你的模型加载逻辑
            # 示例:
            # self.model = YourModel()
            # checkpoint = torch.load(checkpoint_path)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model = self.model.to(device)
            # self.model.eval()
            # self.device = device
            # self.is_loaded = True
            
            print(f"✅ 模型加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.is_loaded = False
            return False
    
    def unload_model(self) -> bool:
        """
        卸载模型，释放显存
        
        必须实现：
        1. 删除模型对象
        2. 清理显存
        3. 设置 self.is_loaded = False
        """
        try:
            # TODO: 实现你的模型卸载逻辑
            # 示例:
            # if self.model is not None:
            #     del self.model
            #     self.model = None
            # 
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # 
            # self.is_loaded = False
            
            print("✅ 模型卸载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型卸载失败: {e}")
            return False
    
    def inference(
        self, 
        image_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        单张图片推理
        
        必须实现：
        1. 加载输入图像
        2. 预处理
        3. 模型推理
        4. 后处理
        5. 保存输出图像
        6. 返回结果字典
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片保存路径
            **kwargs: 其他推理参数（由前端传入）
        
        Returns:
            {
                'success': bool,
                'output_path': str,
                'inference_time': float,
                'message': str,
                'metadata': dict  # 可选的其他信息
            }
        """
        if not self.is_loaded:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        try:
            start_time = time.time()
            
            # TODO: 实现你的推理逻辑
            # 1. 加载图像
            # image = Image.open(image_path)
            
            # 2. 预处理
            # input_tensor = self.preprocess(image)
            
            # 3. 推理
            # with torch.no_grad():
            #     output_tensor = self.model(input_tensor)
            
            # 4. 后处理
            # output_image = self.postprocess(output_tensor)
            
            # 5. 保存
            # output_image.save(output_path)
            
            inference_time = time.time() - start_time
            self.metrics.update(inference_time)
            
            return {
                'success': True,
                'output_path': output_path,
                'inference_time': inference_time,
                'message': '推理成功',
                'metadata': {
                    # 添加你想返回的其他信息
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'推理失败: {str(e)}'
            }
    
    def batch_inference(
        self,
        image_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量推理
        
        可以直接循环调用 inference()，或者实现批量处理以提高效率
        """
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
            'message': f'批量推理完成: {succeeded}/{len(image_paths)} 成功'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        返回模型的基本信息，用于前端展示
        """
        info = {
            'name': '你的模型名称',
            'version': '1.0',
            'description': '模型描述',
            'input_size': (512, 512),  # (H, W)
            'output_size': (512, 512),
            'supported_formats': ['.png', '.jpg', '.jpeg'],
            'default_params': {
                # 默认推理参数
                'param1': value1,
                'param2': value2,
            },
            'custom_fields': {
                # 自定义字段
            }
        }
        
        # 如果模型已加载，添加参数量等信息
        if self.is_loaded and self.model is not None:
            # info['parameters'] = sum(p.numel() for p in self.model.parameters())
            info['performance'] = self.metrics.get_stats()
        
        return info
    
    # ========================================================================
    # 可选的辅助方法
    # ========================================================================
    
    def preprocess(self, image: np.ndarray) -> Any:
        """
        预处理（可选重写）
        
        将 PIL Image 或 numpy array 转换为模型输入格式
        """
        # TODO: 实现你的预处理逻辑
        return image
    
    def postprocess(self, output: Any) -> np.ndarray:
        """
        后处理（可选重写）
        
        将模型输出转换为可保存的图像格式
        """
        # TODO: 实现你的后处理逻辑
        return output


# ============================================================================
# 插件测试代码（可选）
# ============================================================================
if __name__ == "__main__":
    # 测试你的插件
    config = {
        'checkpoint_path': '/path/to/your/checkpoint.pth',
        'device': 'cuda:0',
        # 添加其他配置
    }
    
    plugin = PluginTemplate('my_plugin', config)
    
    if plugin.load_model(config['checkpoint_path']):
        print("\n模型信息:")
        import json
        print(json.dumps(plugin.get_model_info(), indent=2, ensure_ascii=False))
        
        # 测试推理
        test_image = '/path/to/test/image.png'
        result = plugin.inference(
            image_path=test_image,
            output_path='/tmp/test_output.png'
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        plugin.unload_model()

