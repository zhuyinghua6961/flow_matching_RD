"""
FastAPI主入口
"""
import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 自动定位到项目根目录
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent  # 从 webui/backend 回到项目根目录

# 如果当前目录是 webui/backend，切换到项目根目录
if current_dir.name == 'backend' and current_dir.parent.name == 'webui':
    os.chdir(project_root)
    print(f"✓ 自动切换工作目录到项目根目录: {project_root}")

from config import (
    CORS_ORIGINS, LOG_LEVEL, LOG_FORMAT,
    OUTPUT_DIR, PLUGINS_DIR
)
from core import ModelManager
from api import inference_router, model_management_router

# 配置日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Sim2Real推理WebUI",
    description="通用的Sim2Real模型推理框架，支持插件化扩展",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（用于前端访问生成的图片）
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 挂载插件目录（用于下载插件模板）
app.mount("/api/static", StaticFiles(directory=str(PLUGINS_DIR)), name="plugins")

# 创建全局模型管理器
model_manager = ModelManager(plugins_dir=str(PLUGINS_DIR))

# 注册路由
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])
app.include_router(model_management_router, prefix="/api/models", tags=["models"])


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("=" * 70)
    logger.info("Sim2Real推理WebUI启动")
    logger.info("=" * 70)
    logger.info(f"插件目录: {PLUGINS_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # ====================================================================
    # 【方式1】代码注册（可选）：在此处注册插件，启动即可用
    # ====================================================================
    # from plugins.flow_matching_v2_plugin import FlowMatchingV2Plugin
    # 
    # model_manager.register_plugin(
    #     plugin_name='flow_matching_v2',
    #     plugin_class=FlowMatchingV2Plugin,
    #     config={
    #         'checkpoint_path': '/home/user/桌面/flow_matching_RD/outputs_v2/checkpoints/best_model.pth',
    #         'device': 'cuda:0',
    #         'base_channels': 64,
    #         'channel_mult': (1, 2, 4, 8),
    #         'attention_levels': (),
    #         'image_size': (512, 512)
    #     }
    # )
    # logger.info("插件注册完成: flow_matching_v2")
    
    # ====================================================================
    # 【方式2】动态注册（当前模式）：通过前端UI上传插件并注册
    # ====================================================================
    logger.info("动态注册模式：请通过前端UI上传并注册插件")
    logger.info("点击右上角的 [插件管理] 按钮开始")
    
    # ====================================================================
    # 【方式3】自动扫描trained_models目录（新功能）
    # ====================================================================
    logger.info("=" * 70)
    logger.info("正在扫描trained_models目录...")
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    # 检查必要的路径
    trained_models_path = os.path.join(os.getcwd(), "trained_models")
    config_path = os.path.join(os.getcwd(), "config_v2.yaml")
    
    logger.info(f"trained_models路径: {trained_models_path}")
    logger.info(f"trained_models存在: {os.path.exists(trained_models_path)}")
    logger.info(f"config_v2.yaml存在: {os.path.exists(config_path)}")
    
    # 扫描功能将通过API调用，这里保存扫描结果到app.state
    try:
        models = model_manager.scan_trained_models(base_dir="trained_models")
        app.state.scanned_models = models
        logger.info(f"✓ 扫描完成，发现 {len(models)} 个模型")
        if models:
            logger.info("可用模型:")
            for model in models[:5]:  # 只显示前5个
                logger.info(f"  - {model['id']} (Epoch: {model['epoch']}, Loss: {model['val_loss']})")
            if len(models) > 5:
                logger.info(f"  ... 还有 {len(models) - 5} 个模型")
        else:
            logger.warning("⚠ 未找到任何模型！")
            logger.info("请确保:")
            logger.info("  1. trained_models/ 目录存在")
            logger.info("  2. 子目录包含 checkpoints/ 文件夹")
            logger.info("  3. checkpoints/ 下有 .pth 文件")
    except Exception as e:
        logger.error(f"✗ 扫描trained_models目录失败: {e}", exc_info=True)
        app.state.scanned_models = []
    
    logger.info("=" * 70)
    logger.info("服务器启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("正在关闭服务器...")
    
    # 卸载所有已加载的模型
    for plugin_name in list(model_manager.plugins.keys()):
        if model_manager.plugins[plugin_name].is_loaded:
            logger.info(f"卸载插件: {plugin_name}")
            model_manager.unload_model(plugin_name)
    
    logger.info("服务器已关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Sim2Real推理WebUI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "plugins_count": len(model_manager.plugins),
        "current_plugin": model_manager.current_plugin
    }


# 将model_manager挂载到app.state，供路由使用
app.state.model_manager = model_manager


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    
    # 注意：reload=True 会监控文件变化，导致大量 "change detected" 日志
    # 如果不需要修改代码，建议设置为 False
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,  # 关闭自动重载，避免干扰
        log_level=LOG_LEVEL.lower()
    )

