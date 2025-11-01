# ⚡ 快速开始指南

5分钟快速部署并运行 Sim2Real WebUI！

---

## 📋 **环境要求**

- Python 3.8+
- Node.js 16+
- CUDA 11.0+ (推荐，用于GPU加速)

---

## 🚀 **一键启动**

### 步骤 1: 安装后端依赖

```bash
cd webui/backend
pip install -r requirements.txt
```

### 步骤 2: 注册你的模型插件

编辑 `webui/backend/main.py`，在 `startup_event` 中添加：

```python
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("Sim2Real推理WebUI启动")
    logger.info("=" * 70)
    
    # ✅ 在这里注册你的插件
    from plugins.flow_matching_v2_plugin import FlowMatchingV2Plugin
    
    model_manager.register_plugin(
        plugin_name='flow_matching_v2',
        plugin_class=FlowMatchingV2Plugin,
        config={
            'checkpoint_path': '/home/user/桌面/flow_matching_RD/outputs_v2/checkpoints/  checkpoint_epoch_49.pth',
            'device': 'cuda:0',
            'base_channels': 64,
            'channel_mult': (1, 2, 4, 8),
            'attention_levels': (),
            'image_size': (512, 512)
        }
    )
    
    logger.info("插件注册完成")
```

**重要**: 修改 `checkpoint_path` 为你的实际模型路径！

### 步骤 3: 启动后端

```bash
cd webui/backend
python main.py
```

看到以下输出即成功：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
======================================================================
Sim2Real推理WebUI启动
======================================================================
插件目录: /path/to/plugins
输出目录: /path/to/outputs
插件注册完成
服务器启动完成
```

### 步骤 4: 安装前端依赖

**新开一个终端**：

```bash
cd webui/frontend
npm install
```

### 步骤 5: 启动前端

```bash
npm run dev
```

看到：

```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/
```

### 步骤 6: 打开浏览器

访问 `http://localhost:5173`

---

## 🎯 **开始推理**

### 1. 加载模型

1. 在右上角的下拉框中选择插件（如 `flow_matching_v2`）
2. 点击"加载模型"按钮
3. 确认检查点路径（留空则使用注册时的路径）
4. 选择设备（`cuda:0`, `cuda:1`, 或 `cpu`）
5. 点击"确定"

等待几秒钟，看到"模型加载成功"提示即可。

### 2. 上传图片

- 拖拽图片到上传区域
- 或点击上传区域选择文件
- 支持格式: PNG, JPG, JPEG, BMP

### 3. 调整参数（可选）

- **ODE步数**: 10-100（默认50）
  - 步数越多，质量越好，但速度越慢
- **设备**: cuda:0 / cuda:1 / cpu

### 4. 开始推理

点击"开始推理"按钮，等待2-5秒（根据图片大小和ODE步数）。

### 5. 查看结果

推理完成后，右侧会并排展示：
- **左图**: 输入图（Sim）
- **右图**: 输出图（Real）

可以点击"下载"按钮保存结果。

---

## 🔧 **配置调优**

### 显存优化

如果推理时遇到 CUDA OOM：

1. **方案1**: 使用另一块GPU
   ```python
   config = {
       'device': 'cuda:1',  # 改用GPU1
       ...
   }
   ```

2. **方案2**: 启用懒加载
   
   编辑 `webui/backend/config.py`：
   ```python
   LAZY_LOAD = True      # 推理时才加载模型
   AUTO_UNLOAD = True    # 推理后自动卸载
   ```

3. **方案3**: 降低图像分辨率
   ```python
   config = {
       'image_size': (256, 256),  # 降低到256x256
       ...
   }
   ```

### 性能优化

1. **减少ODE步数**（牺牲少量质量）
   - 前端滑块调整为 30 或 20

2. **使用混合精度**（如果你的模型支持）
   ```python
   with torch.amp.autocast('cuda'):
       output = model(input)
   ```

---

## 📱 **后台运行**

### 后端后台运行

```bash
cd webui/backend
nohup python main.py > webui.log 2>&1 &

# 查看日志
tail -f webui.log

# 停止服务
pkill -f "python main.py"
```

### 使用 systemd（生产环境推荐）

创建 `/etc/systemd/system/sim2real-webui.service`：

```ini
[Unit]
Description=Sim2Real WebUI Backend
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/webui/backend
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl start sim2real-webui
sudo systemctl enable sim2real-webui  # 开机自启
sudo systemctl status sim2real-webui  # 查看状态
```

---

## 🌐 **生产部署**

### 前端打包

```bash
cd webui/frontend
npm run build
```

生成的静态文件在 `frontend/dist/` 目录。

### Nginx配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/webui/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端API代理
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # 输出图片代理
    location /outputs {
        proxy_pass http://localhost:8000;
    }
}
```

---

## 🐛 **故障排查**

### 问题1: 后端启动失败

**症状**: `ImportError: No module named 'fastapi'`

**解决**: 
```bash
cd webui/backend
pip install -r requirements.txt
```

### 问题2: 前端无法连接后端

**症状**: 前端显示"Network Error"

**解决**:
1. 确认后端正在运行（访问 `http://localhost:8000/health`）
2. 检查 `vite.config.js` 中的代理配置
3. 检查防火墙是否阻止了8000端口

### 问题3: 插件注册失败

**症状**: "插件 xxx 注册失败"

**解决**:
1. 检查插件类是否继承自 `InferenceInterface`
2. 检查配置参数是否完整
3. 运行插件的测试代码进行调试

### 问题4: 推理失败

**症状**: "推理失败: CUDA out of memory"

**解决**:
1. 确认GPU有足够显存（至少1-2GB空闲）
2. 使用 `nvidia-smi` 查看显存占用
3. 尝试使用另一块GPU或CPU

### 问题5: 图片上传失败

**症状**: "上传失败: 文件格式不支持"

**解决**:
1. 确认图片格式为 PNG, JPG, JPEG, 或 BMP
2. 确认文件大小 < 50MB

---

## 📞 **获取帮助**

- 查看完整文档: [`README.md`](./README.md)
- 插件开发指南: [`PLUGIN_GUIDE.md`](./PLUGIN_GUIDE.md)
- API文档: `http://localhost:8000/docs`

---

## ✅ **下一步**

- [ ] 开发自己的推理插件
- [ ] 添加更多推理参数
- [ ] 批量推理
- [ ] 配置多GPU推理

---

**祝使用愉快！🎉**

