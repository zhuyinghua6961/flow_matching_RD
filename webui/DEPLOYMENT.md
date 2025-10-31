# 🚀 部署指南

本文档提供 WebUI 的详细部署说明。

---

## 📌 **部署架构**

```
┌─────────────┐      HTTP      ┌─────────────┐      API      ┌─────────────┐
│   Browser   │ ────────────> │    Nginx    │ ───────────> │   FastAPI   │
│  (前端UI)    │   (5173/80)   │  (反向代理)  │   (8000)     │   (后端)     │
└─────────────┘               └─────────────┘               └─────────────┘
                                                                     │
                                                                     ↓
                                                             ┌─────────────┐
                                                             │  GPU/Model  │
                                                             │   (推理)     │
                                                             └─────────────┘
```

---

## 🔧 **环境准备**

### 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 7+ / macOS
- **内存**: 8GB+ (推荐16GB+)
- **GPU**: NVIDIA GPU (推荐，CUDA 11.0+)
- **磁盘**: 20GB+ 可用空间

### 软件依赖

```bash
# Python 3.8+
python3 --version

# Node.js 16+
node --version
npm --version

# CUDA (可选，用于GPU加速)
nvcc --version
nvidia-smi
```

---

## 📦 **安装步骤**

### 1. 克隆项目

```bash
cd /home/user/桌面
git clone <your-repo-url> flow_matching_RD
cd flow_matching_RD/webui
```

### 2. 安装后端依赖

```bash
cd backend

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 如果使用PyTorch，还需要安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装前端依赖

```bash
cd ../frontend
npm install
```

---

## 🎯 **开发环境部署**

### 后端（终端1）

```bash
cd webui/backend
python main.py
```

访问 `http://localhost:8000/docs` 查看API文档。

### 前端（终端2）

```bash
cd webui/frontend
npm run dev
```

访问 `http://localhost:5173` 打开WebUI。

---

## 🌐 **生产环境部署**

### 方式1: 使用 Supervisor + Nginx

#### Step 1: 构建前端

```bash
cd webui/frontend
npm run build
```

生成的文件在 `dist/` 目录。

#### Step 2: 配置 Supervisor（后端进程管理）

安装 Supervisor:

```bash
sudo apt install supervisor  # Ubuntu/Debian
# 或
sudo yum install supervisor  # CentOS
```

创建配置文件 `/etc/supervisor/conf.d/sim2real-webui.conf`:

```ini
[program:sim2real-webui]
command=/path/to/venv/bin/python main.py
directory=/path/to/webui/backend
user=your_username
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/sim2real-webui.log
environment=CUDA_VISIBLE_DEVICES="0"
```

启动服务:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sim2real-webui

# 查看状态
sudo supervisorctl status
```

#### Step 3: 配置 Nginx

安装 Nginx:

```bash
sudo apt install nginx  # Ubuntu/Debian
```

创建配置文件 `/etc/nginx/sites-available/sim2real-webui`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 改为你的域名或IP

    # 前端静态文件
    location / {
        root /path/to/webui/frontend/dist;
        try_files $uri $uri/ /index.html;
        
        # 缓存静态资源
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # 后端API代理
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
        
        # 增加超时时间（推理可能较慢）
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }

    # 输出图片代理
    location /outputs {
        proxy_pass http://127.0.0.1:8000;
        expires 1h;
    }

    # 上传文件大小限制
    client_max_body_size 50M;
}
```

启用配置:

```bash
sudo ln -s /etc/nginx/sites-available/sim2real-webui /etc/nginx/sites-enabled/
sudo nginx -t  # 测试配置
sudo systemctl restart nginx
```

#### Step 4: 配置防火墙

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp  # 如果使用HTTPS
sudo ufw reload
```

#### Step 5: 配置HTTPS（可选，推荐）

使用 Let's Encrypt:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### 方式2: 使用 Docker（推荐）

#### Step 1: 创建 Dockerfile

`webui/backend/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建必要目录
RUN mkdir -p uploads outputs plugins

EXPOSE 8000

CMD ["python", "main.py"]
```

`webui/frontend/Dockerfile`:

```dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Step 2: 创建 docker-compose.yml

`webui/docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/uploads:/app/uploads
      - ./backend/outputs:/app/outputs
      - ./backend/plugins:/app/plugins
      - /path/to/your/models:/models:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  uploads:
  outputs:
```

#### Step 3: 启动服务

```bash
cd webui
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 🔒 **安全配置**

### 1. API认证（可选）

在 `backend/main.py` 中添加认证中间件:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your_secret_token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# 在需要保护的路由中添加依赖
@router.post("/infer", dependencies=[Depends(verify_token)])
async def infer_image(...):
    ...
```

### 2. CORS配置

编辑 `backend/config.py`:

```python
CORS_ORIGINS = [
    "https://your-domain.com",  # 生产域名
]
```

### 3. 文件上传限制

```python
# config.py
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 限制为10MB
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
```

---

## 📊 **监控与日志**

### 日志配置

编辑 `backend/config.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sim2real-webui.log'),
        logging.StreamHandler()
    ]
)
```

### 性能监控

使用 Prometheus + Grafana:

```python
# 安装依赖
pip install prometheus-fastapi-instrumentator

# 在 main.py 中添加
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## 🔄 **更新与备份**

### 更新代码

```bash
cd /path/to/webui
git pull

# 更新后端依赖
cd backend
pip install -r requirements.txt

# 重新构建前端
cd ../frontend
npm install
npm run build

# 重启服务
sudo supervisorctl restart sim2real-webui
sudo systemctl restart nginx
```

### 备份数据

```bash
# 备份上传和输出文件
tar -czf webui-backup-$(date +%Y%m%d).tar.gz \
    webui/backend/uploads \
    webui/backend/outputs \
    webui/backend/plugins
```

---

## 🐛 **故障排查**

### 查看日志

```bash
# Supervisor日志
sudo tail -f /var/log/sim2real-webui.log

# Nginx日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Docker日志
docker-compose logs -f backend
```

### 常见问题

1. **502 Bad Gateway**: 检查后端是否正常运行
2. **CUDA OOM**: 降低batch size或使用CPU
3. **权限错误**: 检查文件夹权限 `chmod -R 755 uploads outputs`

---

## ✅ **检查清单**

部署完成后，检查以下项：

- [ ] 后端API正常响应 (`curl http://localhost:8000/health`)
- [ ] 前端页面可访问
- [ ] 模型插件已注册
- [ ] 图片上传功能正常
- [ ] 推理功能正常
- [ ] 日志记录正常
- [ ] SSL证书有效（如果使用HTTPS）
- [ ] 备份策略已配置

---

**部署完成！🎉**

