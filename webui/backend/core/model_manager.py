"""
模型管理器

负责管理所有已注册的推理插件，支持：
1. 动态加载/卸载插件
2. 模型切换
3. 插件状态管理
4. 自动扫描trained_models目录
"""
import importlib
import importlib.util
import sys
import os
import glob
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional, List, Any
import logging

from .inference_interface import InferenceInterface

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        """
        初始化模型管理器
        
        Args:
            plugins_dir: 插件目录路径
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, InferenceInterface] = {}
        self.current_plugin: Optional[str] = None
        
        # 确保插件目录存在
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager初始化，插件目录: {self.plugins_dir}")
    
    def register_plugin(
        self, 
        plugin_name: str, 
        plugin_class: type,
        config: Dict
    ) -> bool:
        """
        注册插件
        
        Args:
            plugin_name: 插件名称（唯一标识）
            plugin_class: 插件类（继承自InferenceInterface）
            config: 插件配置
            
        Returns:
            bool: 注册成功返回True
        """
        try:
            # 检查是否继承自InferenceInterface
            if not issubclass(plugin_class, InferenceInterface):
                logger.error(f"插件 {plugin_name} 必须继承自 InferenceInterface")
                return False
            
            # 实例化插件
            plugin_instance = plugin_class(plugin_name, config)
            
            # 验证配置
            if not plugin_instance.validate_config():
                logger.error(f"插件 {plugin_name} 配置验证失败")
                return False
            
            # 注册
            self.plugins[plugin_name] = plugin_instance
            logger.info(f"插件 {plugin_name} 注册成功")
            
            # 如果是第一个插件，设置为当前插件
            if self.current_plugin is None:
                self.current_plugin = plugin_name
                
            return True
            
        except Exception as e:
            logger.error(f"注册插件 {plugin_name} 失败: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        注销插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 注销成功返回True
        """
        if plugin_name not in self.plugins:
            logger.warning(f"插件 {plugin_name} 不存在")
            return False
        
        # 如果模型已加载，先卸载
        plugin = self.plugins[plugin_name]
        if plugin.is_loaded:
            plugin.unload_model()
        
        # 删除插件
        del self.plugins[plugin_name]
        logger.info(f"插件 {plugin_name} 已注销")
        
        # 如果是当前插件，清空当前插件
        if self.current_plugin == plugin_name:
            self.current_plugin = None
            
        return True
    
    def load_plugin_from_file(
        self,
        plugin_file: str,
        plugin_class_name: str,
        plugin_name: str,
        config: Dict
    ) -> bool:
        """
        从文件动态加载插件
        
        Args:
            plugin_file: 插件文件路径（.py文件），支持绝对路径和相对路径
            plugin_class_name: 插件类名
            plugin_name: 插件注册名称
            config: 插件配置
            
        Returns:
            bool: 加载成功返回True
        """
        try:
            # 处理路径
            plugin_path = Path(plugin_file)
            
            # 如果是相对路径，相对于插件目录
            if not plugin_path.is_absolute():
                plugin_path = self.plugins_dir / plugin_path
            
            # 检查文件是否存在
            if not plugin_path.exists():
                logger.error(f"插件文件不存在: {plugin_path}")
                return False
            
            if not plugin_path.is_file():
                logger.error(f"不是有效的文件: {plugin_path}")
                return False
            
            if plugin_path.suffix != '.py':
                logger.error(f"不是Python文件: {plugin_path}")
                return False
            
            # 导入模块
            spec = importlib.util.spec_from_file_location(
                f"dynamic_plugin_{plugin_name}", 
                str(plugin_path)
            )
            
            if spec is None:
                logger.error(f"无法创建模块规范: {plugin_path}")
                return False
            
            if spec.loader is None:
                logger.error(f"无法获取模块加载器: {plugin_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"dynamic_plugin_{plugin_name}"] = module
            spec.loader.exec_module(module)
            
            # 获取类
            if not hasattr(module, plugin_class_name):
                logger.error(f"模块中不存在类 '{plugin_class_name}': {plugin_path}")
                return False
            
            plugin_class = getattr(module, plugin_class_name)
            
            # 注册插件
            return self.register_plugin(plugin_name, plugin_class, config)
            
        except Exception as e:
            logger.error(f"从文件 {plugin_file} 加载插件失败: {e}", exc_info=True)
            return False
    
    def switch_plugin(self, plugin_name: str) -> bool:
        """
        切换当前使用的插件
        
        Args:
            plugin_name: 目标插件名称
            
        Returns:
            bool: 切换成功返回True
        """
        if plugin_name not in self.plugins:
            logger.error(f"插件 {plugin_name} 不存在")
            return False
        
        self.current_plugin = plugin_name
        logger.info(f"已切换到插件: {plugin_name}")
        return True
    
    def get_current_plugin(self) -> Optional[InferenceInterface]:
        """
        获取当前插件实例
        
        Returns:
            当前插件实例，如果没有则返回None
        """
        if self.current_plugin is None:
            return None
        return self.plugins.get(self.current_plugin)
    
    def get_plugin(self, plugin_name: str) -> Optional[InferenceInterface]:
        """
        获取指定插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件实例，如果不存在则返回None
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict]:
        """
        列出所有已注册的插件
        
        Returns:
            插件信息列表
        """
        result = []
        for name, plugin in self.plugins.items():
            info = {
                'name': name,
                'is_loaded': plugin.is_loaded,
                'is_current': name == self.current_plugin,
            }
            
            # 如果模型已加载，添加模型信息
            if plugin.is_loaded:
                try:
                    info['model_info'] = plugin.get_model_info()
                except:
                    pass
            
            result.append(info)
        
        return result
    
    def load_model(
        self, 
        plugin_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0'
    ) -> bool:
        """
        加载模型
        
        Args:
            plugin_name: 插件名称（None则使用当前插件）
            checkpoint_path: 检查点路径（None则使用配置中的路径）
            device: 设备
            
        Returns:
            bool: 加载成功返回True
        """
        logger.info(f"[ModelManager] 开始加载模型: plugin_name={plugin_name}, device={device}")
        
        # 确定目标插件
        if plugin_name is None:
            plugin = self.get_current_plugin()
            plugin_name = self.current_plugin
        else:
            plugin = self.get_plugin(plugin_name)
        
        logger.info(f"[ModelManager] 插件实例: {plugin}, plugin_name={plugin_name}")
        
        if plugin is None:
            logger.error(f"插件 {plugin_name} 不存在")
            logger.error(f"[ModelManager] 当前已注册的插件: {list(self.plugins.keys())}")
            return False
        
        # 如果已加载，先卸载
        if plugin.is_loaded:
            logger.info(f"插件 {plugin_name} 已加载，先卸载")
            plugin.unload_model()
        
        # 确定检查点路径
        if checkpoint_path is None:
            checkpoint_path = plugin.config.get('checkpoint_path')
        
        logger.info(f"[ModelManager] checkpoint_path: {checkpoint_path}")
        logger.info(f"[ModelManager] 插件配置: {plugin.config}")
        
        if checkpoint_path is None:
            logger.error(f"插件 {plugin_name} 未指定检查点路径")
            return False
        
        # 加载模型
        try:
            logger.info(f"[ModelManager] 调用插件的 load_model 方法...")
            success = plugin.load_model(checkpoint_path, device)
            if success:
                logger.info(f"插件 {plugin_name} 模型加载成功")
            else:
                logger.error(f"插件 {plugin_name} 模型加载失败")
            return success
        except Exception as e:
            logger.error(f"插件 {plugin_name} 加载模型异常: {e}", exc_info=True)
            return False
    
    def unload_model(self, plugin_name: Optional[str] = None) -> bool:
        """
        卸载模型
        
        Args:
            plugin_name: 插件名称（None则使用当前插件）
            
        Returns:
            bool: 卸载成功返回True
        """
        if plugin_name is None:
            plugin = self.get_current_plugin()
            plugin_name = self.current_plugin
        else:
            plugin = self.get_plugin(plugin_name)
        
        if plugin is None:
            logger.error(f"插件 {plugin_name} 不存在")
            return False
        
        try:
            success = plugin.unload_model()
            if success:
                logger.info(f"插件 {plugin_name} 模型卸载成功")
            return success
        except Exception as e:
            logger.error(f"插件 {plugin_name} 卸载模型异常: {e}")
            return False
    
    def inference(
        self,
        image_path: str,
        output_path: str,
        plugin_name: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        执行推理
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            plugin_name: 插件名称（None则使用当前插件）
            **kwargs: 推理参数
            
        Returns:
            推理结果字典
        """
        if plugin_name is None:
            plugin = self.get_current_plugin()
            plugin_name = self.current_plugin
        else:
            plugin = self.get_plugin(plugin_name)
        
        if plugin is None:
            return {
                'success': False,
                'message': f'插件 {plugin_name} 不存在'
            }
        
        if not plugin.is_loaded:
            return {
                'success': False,
                'message': f'插件 {plugin_name} 模型未加载'
            }
        
        try:
            return plugin.inference(image_path, output_path, **kwargs)
        except Exception as e:
            logger.error(f"推理异常: {e}")
            return {
                'success': False,
                'message': f'推理异常: {str(e)}'
            }
    
    def batch_inference(
        self,
        image_paths: List[str],
        output_dir: str,
        plugin_name: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        批量推理
        
        Args:
            image_paths: 输入图片路径列表
            output_dir: 输出目录
            plugin_name: 插件名称（None则使用当前插件）
            **kwargs: 推理参数
            
        Returns:
            批量推理结果字典
        """
        if plugin_name is None:
            plugin = self.get_current_plugin()
            plugin_name = self.current_plugin
        else:
            plugin = self.get_plugin(plugin_name)
        
        if plugin is None:
            return {
                'success': False,
                'message': f'插件 {plugin_name} 不存在'
            }
        
        if not plugin.is_loaded:
            return {
                'success': False,
                'message': f'插件 {plugin_name} 模型未加载'
            }
        
        try:
            return plugin.batch_inference(image_paths, output_dir, **kwargs)
        except Exception as e:
            logger.error(f"批量推理异常: {e}")
            return {
                'success': False,
                'message': f'批量推理异常: {str(e)}'
            }
    
    def scan_trained_models(
        self, 
        base_dir: str = "trained_models",
        global_config_path: str = "config_v2.yaml",
        recursive: bool = True
    ) -> List[Dict]:
        """
        扫描trained_models目录，自动发现所有可用模型
        
        约定目录结构:
          项目根目录/
          ├── config_v2.yaml              ← 全局配置（所有模型共用）
          └── trained_models/
              ├── project1/
              │   └── checkpoints/
              │       ├── best_model.pth
              │       └── final_model.pth
              └── history_models/
                  └── project2/
                      └── checkpoints/
                          └── model.pth
        
        Args:
            base_dir: 基础目录路径
            global_config_path: 全局配置文件路径（相对于项目根目录）
            recursive: 是否递归扫描子目录
            
        Returns:
            模型信息列表，每个模型包含:
            - id: 唯一标识 (project/model_name)
            - name: 模型文件名
            - project: 项目名称
            - path: 模型文件绝对路径
            - config_path: 配置文件路径
            - config: 配置字典
            - has_config: 是否有配置文件
            - epoch: epoch数（从checkpoint读取）
            - val_loss: 验证loss（从checkpoint读取）
        """
        models = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.warning(f"训练模型目录不存在: {base_dir}")
            return models
        
        logger.info(f"开始扫描训练模型目录: {base_dir}")
        
        # 尝试加载全局配置文件（从项目根目录）
        global_config_data = None
        global_config_full_path = None
        
        # 尝试多个可能的配置文件名
        possible_config_names = ['config_v2.yaml', 'config.yaml', 'config_v2.yml', 'config.yml']
        for config_name in possible_config_names:
            config_path = Path(global_config_path).parent / config_name if '/' in global_config_path else Path(config_name)
            if config_path.exists():
                try:
                    global_config_data = self._load_config(str(config_path))
                    global_config_full_path = str(config_path.absolute())
                    logger.info(f"✓ 找到全局配置文件: {global_config_full_path}")
                    break
                except Exception as e:
                    logger.warning(f"加载全局配置文件失败 {config_path}: {e}")
        
        if global_config_data is None:
            logger.warning(f"⚠ 未找到全局配置文件，将使用默认参数")
        
        # 递归查找所有 checkpoints 目录
        for root, dirs, files in os.walk(base_path):
            if "checkpoints" in dirs:
                checkpoint_dir = os.path.join(root, "checkpoints")
                project_dir = root
                project_name = os.path.relpath(root, base_path)
                
                # 优先使用全局配置，如果本地有配置文件则覆盖
                config_path = global_config_full_path
                config_data = global_config_data
                
                # 检查是否有本地配置文件（可选，用于覆盖全局配置）
                local_config_path = self._find_config_file(project_dir)
                if local_config_path:
                    try:
                        local_config_data = self._load_config(local_config_path)
                        config_path = local_config_path
                        config_data = local_config_data
                        logger.debug(f"使用本地配置文件: {local_config_path}")
                    except Exception as e:
                        logger.warning(f"加载本地配置文件失败 {local_config_path}: {e}")
                
                # 扫描所有 .pth 文件
                pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                
                for pth_file in pth_files:
                    try:
                        # 读取checkpoint信息
                        checkpoint_info = self._load_checkpoint_info(pth_file)
                        
                        # 生成模型ID
                        model_name = os.path.basename(pth_file)
                        model_id = f"{project_name}/{model_name}"
                        
                        model_info = {
                            'id': model_id,
                            'name': model_name,
                            'project': project_name,
                            'path': os.path.abspath(pth_file),
                            'config_path': config_path,
                            'config': config_data,
                            'has_config': config_data is not None,
                            'epoch': checkpoint_info.get('epoch', 'unknown'),
                            'val_loss': checkpoint_info.get('val_loss', 'unknown'),
                        }
                        
                        models.append(model_info)
                        logger.debug(f"发现模型: {model_id}")
                        
                    except Exception as e:
                        logger.warning(f"处理模型文件失败 {pth_file}: {e}")
        
        logger.info(f"扫描完成，找到 {len(models)} 个模型")
        return models
    
    def _find_config_file(self, directory: str) -> Optional[str]:
        """
        在目录及父目录中查找配置文件
        
        Args:
            directory: 搜索目录
            
        Returns:
            配置文件路径，未找到返回None
        """
        config_names = ['config_v2.yaml', 'config.yaml', 'config_v2.yml', 'config.yml']
        
        # 在当前目录查找
        for name in config_names:
            config_path = os.path.join(directory, name)
            if os.path.exists(config_path):
                return config_path
        
        return None
    
    def _load_config(self, config_path: str) -> Dict:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """
        从checkpoint文件中提取信息
        
        Args:
            checkpoint_path: checkpoint文件路径
            
        Returns:
            包含epoch和val_loss等信息的字典
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'unknown'))
            }
        except Exception as e:
            logger.debug(f"无法读取checkpoint信息 {checkpoint_path}: {e}")
            return {}
    
    def auto_register_from_config(
        self,
        model_info: Dict,
        plugin_template_path: str = "flow_matching_v2_plugin.py",
        device: str = 'cuda:0'
    ) -> bool:
        """
        根据模型信息和配置自动注册插件
        
        Args:
            model_info: 模型信息字典（来自scan_trained_models）
            plugin_template_path: 插件模板路径
            device: 设备
            
        Returns:
            bool: 注册成功返回True
        """
        try:
            model_id = model_info['id']
            model_path = model_info['path']
            config_data = model_info.get('config')
            
            # 构建插件配置
            plugin_config = {
                'checkpoint_path': model_path,
                'device': device,
            }
            
            # 从训练配置中提取参数
            if config_data:
                # 提取normalize参数
                if 'data' in config_data:
                    data_config = config_data['data']
                    plugin_config['normalize_mean'] = data_config.get('normalize_mean', 0.35)
                    plugin_config['normalize_std'] = data_config.get('normalize_std', 0.06)
                
                # 提取推理参数
                if 'inference' in config_data:
                    inference_config = config_data['inference']
                    plugin_config['ode_steps'] = inference_config.get('ode_steps', 50)
                    plugin_config['ode_method'] = inference_config.get('ode_method', 'euler')
                
                # 提取模型架构参数
                if 'model' in config_data:
                    plugin_config['model_config'] = config_data['model']
            else:
                # 使用默认参数
                logger.warning(f"模型 {model_id} 没有配置文件，使用默认参数")
                plugin_config['normalize_mean'] = 0.35
                plugin_config['normalize_std'] = 0.06
                plugin_config['ode_steps'] = 50
                plugin_config['ode_method'] = 'euler'
            
            # 加载插件类
            plugin_name = model_id.replace('/', '_').replace('.pth', '')
            
            # 使用通用的FlowMatchingV2Plugin
            success = self.load_plugin_from_file(
                plugin_file=plugin_template_path,
                plugin_class_name='FlowMatchingV2Plugin',
                plugin_name=plugin_name,
                config=plugin_config
            )
            
            if success:
                logger.info(f"自动注册模型成功: {model_id}")
            else:
                logger.error(f"自动注册模型失败: {model_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"自动注册模型异常: {e}", exc_info=True)
            return False

