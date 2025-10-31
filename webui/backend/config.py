"""
后端配置文件
"""
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
PLUGINS_DIR = BASE_DIR / "plugins"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000

# 文件上传配置
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
ALLOWED_PLUGIN_EXTENSIONS = {'.py'}

# CORS配置
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite默认开发端口
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 推理配置
DEFAULT_DEVICE = "cuda:0"  # 默认推理设备
LAZY_LOAD = True  # 是否懒加载（推理时才加载模型）
AUTO_UNLOAD = False  # 推理完成后是否自动卸载模型

# 任务队列配置（可选，用于异步推理）
ENABLE_TASK_QUEUE = False
MAX_WORKERS = 2

