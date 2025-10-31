"""
模型管理器

负责管理所有已注册的推理插件，支持：
1. 动态加载/卸载插件
2. 模型切换
3. 插件状态管理
"""
import importlib
import importlib.util
import sys
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
        # 确定目标插件
        if plugin_name is None:
            plugin = self.get_current_plugin()
            plugin_name = self.current_plugin
        else:
            plugin = self.get_plugin(plugin_name)
        
        if plugin is None:
            logger.error(f"插件 {plugin_name} 不存在")
            return False
        
        # 如果已加载，先卸载
        if plugin.is_loaded:
            logger.info(f"插件 {plugin_name} 已加载，先卸载")
            plugin.unload_model()
        
        # 确定检查点路径
        if checkpoint_path is None:
            checkpoint_path = plugin.config.get('checkpoint_path')
        
        if checkpoint_path is None:
            logger.error(f"插件 {plugin_name} 未指定检查点路径")
            return False
        
        # 加载模型
        try:
            success = plugin.load_model(checkpoint_path, device)
            if success:
                logger.info(f"插件 {plugin_name} 模型加载成功")
            else:
                logger.error(f"插件 {plugin_name} 模型加载失败")
            return success
        except Exception as e:
            logger.error(f"插件 {plugin_name} 加载模型异常: {e}")
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

