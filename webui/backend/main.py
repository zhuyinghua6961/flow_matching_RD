"""
FastAPI主入口
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # 开发模式下自动重载
        log_level=LOG_LEVEL.lower()
    )

