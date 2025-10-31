"""
API路由模块
"""
from .inference import router as inference_router
from .model_management import router as model_management_router

__all__ = ['inference_router', 'model_management_router']

