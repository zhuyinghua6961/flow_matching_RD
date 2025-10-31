"""
模型管理相关API
"""
import logging
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from config import PLUGINS_DIR, ALLOWED_PLUGIN_EXTENSIONS

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterPluginRequest(BaseModel):
    """注册插件请求"""
    plugin_file: str
    plugin_class_name: str
    plugin_name: str
    config: dict


class LoadModelRequest(BaseModel):
    """加载模型请求"""
    plugin_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: str = "cuda:0"


class SwitchPluginRequest(BaseModel):
    """切换插件请求"""
    plugin_name: str


@router.get("/list")
async def list_plugins(request: Request):
    """
    列出所有已注册的插件
    
    Returns:
        {
            'success': bool,
            'plugins': List[dict],
            'current_plugin': str,
            'total': int
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        plugins = model_manager.list_plugins()
        
        return {
            'success': True,
            'plugins': plugins,
            'current_plugin': model_manager.current_plugin,
            'total': len(plugins)
        }
    except Exception as e:
        logger.error(f"列出插件失败: {e}")
        raise HTTPException(status_code=500, detail=f"操作失败: {str(e)}")


@router.post("/upload_plugin")
async def upload_plugin(file: UploadFile = File(...)):
    """
    上传插件文件
    
    Returns:
        {
            'success': bool,
            'file_path': str,
            'file_name': str,
            'message': str
        }
    """
    try:
        # 检查文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_PLUGIN_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的插件格式: {file_ext}"
            )
        
        # 保存文件
        file_path = PLUGINS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"上传插件成功: {file.filename}")
        
        return {
            'success': True,
            'file_path': str(file_path),
            'file_name': file.filename,
            'message': '上传成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传插件失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/register")
async def register_plugin(request: Request, req_body: RegisterPluginRequest):
    """
    注册插件
    
    Args:
        plugin_file: 插件文件路径
        plugin_class_name: 插件类名
        plugin_name: 插件注册名称
        config: 插件配置
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        # 检查插件文件是否存在
        plugin_path = Path(req_body.plugin_file)
        if not plugin_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"插件文件不存在: {req_body.plugin_file}"
            )
        
        # 加载并注册插件
        success = model_manager.load_plugin_from_file(
            plugin_file=str(plugin_path),
            plugin_class_name=req_body.plugin_class_name,
            plugin_name=req_body.plugin_name,
            config=req_body.config
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="插件注册失败"
            )
        
        logger.info(f"插件注册成功: {req_body.plugin_name}")
        
        return {
            'success': True,
            'plugin_name': req_body.plugin_name,
            'message': '注册成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册插件失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@router.post("/load")
async def load_model(request: Request, req_body: LoadModelRequest):
    """
    加载模型
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        success = model_manager.load_model(
            plugin_name=req_body.plugin_name,
            checkpoint_path=req_body.checkpoint_path,
            device=req_body.device
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="模型加载失败")
        
        plugin_name = req_body.plugin_name or model_manager.current_plugin
        logger.info(f"模型加载成功: {plugin_name}")
        
        return {
            'success': True,
            'plugin_name': plugin_name,
            'message': '加载成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载失败: {str(e)}")


@router.post("/unload")
async def unload_model(request: Request, plugin_name: Optional[str] = None):
    """
    卸载模型
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        success = model_manager.unload_model(plugin_name)
        
        if not success:
            raise HTTPException(status_code=500, detail="模型卸载失败")
        
        target_plugin = plugin_name or model_manager.current_plugin
        logger.info(f"模型卸载成功: {target_plugin}")
        
        return {
            'success': True,
            'plugin_name': target_plugin,
            'message': '卸载成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"卸载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"卸载失败: {str(e)}")


@router.post("/switch")
async def switch_plugin(request: Request, req_body: SwitchPluginRequest):
    """
    切换当前使用的插件
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        success = model_manager.switch_plugin(req_body.plugin_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"插件不存在: {req_body.plugin_name}")
        
        logger.info(f"切换插件成功: {req_body.plugin_name}")
        
        return {
            'success': True,
            'plugin_name': req_body.plugin_name,
            'message': '切换成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换插件失败: {e}")
        raise HTTPException(status_code=500, detail=f"切换失败: {str(e)}")


@router.get("/info/{plugin_name}")
async def get_plugin_info(request: Request, plugin_name: str):
    """
    获取插件信息
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'is_loaded': bool,
            'model_info': dict,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        plugin = model_manager.get_plugin(plugin_name)
        
        if plugin is None:
            raise HTTPException(status_code=404, detail=f"插件不存在: {plugin_name}")
        
        result = {
            'success': True,
            'plugin_name': plugin_name,
            'is_loaded': plugin.is_loaded,
            'message': '获取成功'
        }
        
        # 如果模型已加载，获取模型信息
        if plugin.is_loaded:
            try:
                result['model_info'] = plugin.get_model_info()
            except Exception as e:
                logger.warning(f"获取模型信息失败: {e}")
                result['model_info'] = {}
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取插件信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"操作失败: {str(e)}")


@router.delete("/unregister/{plugin_name}")
async def unregister_plugin(request: Request, plugin_name: str):
    """
    注销插件
    
    Returns:
        {
            'success': bool,
            'plugin_name': str,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        success = model_manager.unregister_plugin(plugin_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"插件不存在: {plugin_name}")
        
        logger.info(f"插件注销成功: {plugin_name}")
        
        return {
            'success': True,
            'plugin_name': plugin_name,
            'message': '注销成功'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注销插件失败: {e}")
        raise HTTPException(status_code=500, detail=f"注销失败: {str(e)}")

