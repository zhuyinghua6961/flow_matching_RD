# Sim2Real 推理 WebUI

通用的 Sim2Real 模型推理框架，支持插件化扩展。

---

## 📌 **核心特性**

### ✨ **通用性**
- **插件化架构**：任何 Sim2Real 模型只需实现标准接口即可接入
- **零代码修改**：上传插件即用，无需修改 WebUI 代码
- **多模型管理**：支持同时注册多个模型，动态切换

### 🚀 **高性能**
- **GPU加速**：支持多GPU推理，可指定设备
- **批量处理**：支持批量上传和推理
- **懒加载模式**：可选择推理时才加载模型，节省显存

### 🎨 **易用性**
- **现代化UI**：Vue 3 + Element Plus，响应式设计
- **实时对比**：输入/输出图像并排展示
- **参数可调**：ODE步数、设备等参数在线调节

---

## 📁 **项目结构**

```
webui/
├── backend/                    # 后端 (FastAPI)
│   ├── core/                   # 核心框架
│   │   ├── inference_interface.py  # 推理接口（抽象基类）
│   │   └── model_manager.py        # 模型管理器
│   ├── api/                    # API路由
│   │   ├── inference.py            # 推理API
│   │   └── model_management.py     # 模型管理API
│   ├── plugins/                # 插件目录（用户自定义）
│   │   ├── plugin_template.py      # 插件模板
│   │   └── flow_matching_v2_plugin.py  # 示例插件
│   ├── uploads/                # 上传文件目录
│   ├── outputs/                # 输出文件目录
│   ├── config.py               # 配置文件
│   ├── main.py                 # FastAPI入口
│   └── requirements.txt        # 依赖列表
│
├── frontend/                   # 前端 (Vue 3)
│   ├── src/
│   │   ├── components/         # Vue组件
│   │   │   ├── ImageUpload.vue     # 图片上传
│   │   │   ├── ResultDisplay.vue   # 结果展示
│   │   │   ├── ModelSelector.vue   # 模型选择
│   │   │   └── InferenceParams.vue # 参数配置
│   │   ├── stores/             # Pinia状态管理
│   │   │   ├── model.js            # 模型状态
│   │   │   └── inference.js        # 推理状态
│   │   ├── api/                # API接口
│   │   │   ├── inference.js        # 推理API
│   │   │   └── model.js            # 模型API
│   │   ├── App.vue             # 主组件
│   │   └── main.js             # 入口文件
│   ├── package.json
│   └── vite.config.js
│
└── README.md                   # 本文件
```

---

## 🚀 **快速开始**

### 选择你的启动方式

我们提供**两种插件注册方式**，根据你的需求选择：

| 方式 | 适合场景 | 优点 | 快速开始指南 |
|------|---------|------|------------|
| **方式1：代码注册** | 开发环境、单一模型 | 简单直接，启动即用 | [`QUICKSTART.md`](./QUICKSTART.md) |
| **方式2：动态注册** ⭐ | 生产环境、多用户 | 前端UI操作，无需重启 | [`QUICKSTART_DYNAMIC.md`](./QUICKSTART_DYNAMIC.md) |

---

### 方式1：代码注册（传统方式）

#### 1. 安装依赖
```bash
# 后端
cd webui/backend
pip install -r requirements.txt

# 前端
cd webui/frontend
npm install
```

#### 2. 注册插件

编辑 `backend/main.py`，取消注释代码注册部分：

```python
@app.on_event("startup")
async def startup_event():
    # 取消注释以下代码
    from plugins.flow_matching_v2_plugin import FlowMatchingV2Plugin
    
    model_manager.register_plugin(
        plugin_name='flow_matching_v2',
        plugin_class=FlowMatchingV2Plugin,
        config={
            'checkpoint_path': '/path/to/your/checkpoint.pth',
            'device': 'cuda:0',
            'base_channels': 64,
            'channel_mult': (1, 2, 4, 8),
            'attention_levels': (),
            'image_size': (512, 512)
        }
    )
```

#### 3. 启动服务
```bash
# 后端
cd webui/backend
python main.py

# 前端（新终端）
cd webui/frontend
npm run dev
```

#### 4. 访问
打开浏览器访问 `http://localhost:5173`

---

### 方式2：动态注册（推荐）⭐

#### 1. 安装依赖
```bash
# 后端
cd webui/backend
pip install -r requirements.txt

# 前端
cd webui/frontend
npm install
```

#### 2. 启动服务（无需修改代码）
```bash
# 后端
cd webui/backend
python main.py

# 前端（新终端）
cd webui/frontend
npm run dev
```

#### 3. 在前端UI注册插件

1. 打开浏览器访问 `http://localhost:5173`
2. 点击右上角的 **[插件管理]** 按钮
3. 切换到"注册新插件"标签页
4. 选择"指定路径"，输入：`plugins/flow_matching_v2_plugin.py`
   - 或选择"上传文件"，上传插件文件
5. 填写配置信息
6. 点击"注册插件"

详细步骤见 [`QUICKSTART_DYNAMIC.md`](./QUICKSTART_DYNAMIC.md)

**提示**: 使用"指定路径"模式可以避免上传，直接使用服务器上的文件！

---

### 推理流程

无论使用哪种方式，推理流程相同：

1. 选择插件
2. 加载模型
3. 上传图片
4. 调整参数
5. 开始推理
6. 查看/下载结果

---

## 🔌 **插件开发指南**

详见 [`PLUGIN_GUIDE.md`](./PLUGIN_GUIDE.md)

---

## 📡 **API接口**

### 推理API

- `POST /api/inference/upload` - 上传单张图片
- `POST /api/inference/upload_batch` - 批量上传图片
- `POST /api/inference/infer` - 单张推理
- `POST /api/inference/infer_batch` - 批量推理
- `GET /api/inference/list_uploaded` - 列出已上传图片
- `GET /api/inference/list_outputs` - 列出输出图片

### 模型管理API

- `GET /api/models/list` - 列出所有插件
- `POST /api/models/upload_plugin` - 上传插件文件
- `POST /api/models/register` - 注册插件
- `POST /api/models/load` - 加载模型
- `POST /api/models/unload` - 卸载模型
- `POST /api/models/switch` - 切换插件
- `GET /api/models/info/{plugin_name}` - 获取插件信息
- `DELETE /api/models/unregister/{plugin_name}` - 注销插件

完整API文档: `http://localhost:8000/docs`

---

## ⚙️ **配置说明**

### 后端配置 (`backend/config.py`)

```python
# 路径配置
UPLOAD_DIR = "uploads"      # 上传目录
OUTPUT_DIR = "outputs"      # 输出目录
PLUGINS_DIR = "plugins"     # 插件目录

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000

# 文件上传配置
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# 推理配置
DEFAULT_DEVICE = "cuda:0"   # 默认推理设备
LAZY_LOAD = True            # 懒加载模式
AUTO_UNLOAD = False         # 推理后自动卸载
```

### 前端配置 (`frontend/vite.config.js`)

```javascript
export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // 后端地址
        changeOrigin: true
      }
    }
  }
})
```

---

## 🐛 **常见问题**

### Q: 推理时报"模型未加载"？
**A**: 请先在模型选择器中选择插件并点击"加载模型"。

### Q: 如何同时支持多GPU推理？
**A**: 
- 训练用 `cuda:0`
- WebUI推理用 `cuda:1`
- 或者使用懒加载模式（`LAZY_LOAD=True`），推理时动态加载/卸载

### Q: 推理速度很慢？
**A**: 
1. 确认使用GPU（`device=cuda:0`）
2. 减少ODE步数（如从50降到30）
3. 检查模型是否正确加载到GPU

### Q: 如何添加自定义推理参数？
**A**: 
1. 在插件的 `inference()` 方法中接收 `**kwargs`
2. 在前端 `InferenceParams.vue` 中添加参数输入框
3. 参数会通过 `custom_params` 传递给插件

---

## 📝 **开发计划**

- [ ] 支持WebSocket实时推理进度
- [ ] 添加推理历史记录
- [ ] 支持模型性能对比
- [ ] 添加图像预处理选项
- [ ] 支持视频推理
- [ ] Docker部署

---

## 📄 **许可证**

MIT License

---

## 🙏 **致谢**

- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue 3](https://vuejs.org/)
- [Element Plus](https://element-plus.org/)
- [Pinia](https://pinia.vuejs.org/)

