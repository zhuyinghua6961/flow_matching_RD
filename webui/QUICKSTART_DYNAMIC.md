# ⚡ 快速开始指南 - 动态注册模式

无需修改代码，通过前端UI直接上传和注册插件！

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

### 步骤 2: 启动后端

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
动态注册模式：请通过前端UI上传并注册插件
点击右上角的 [插件管理] 按钮开始
服务器启动完成
```

### 步骤 3: 安装前端依赖

**新开一个终端**：

```bash
cd webui/frontend
npm install
```

### 步骤 4: 启动前端

```bash
npm run dev
```

看到：

```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/
```

### 步骤 5: 打开浏览器

访问 `http://localhost:5173`

---

## 🔌 **动态注册插件**

### 方式A：使用已有的示例插件（推荐）⭐

如果你已经有项目的 V2 模型，可以直接注册现有插件：

1. **打开插件管理**
   - 点击右上角的 **[插件管理]** 按钮

2. **切换到"注册新插件"标签页**

3. **选择插件文件**（两种方式）

   **方式1：指定路径（推荐，无需上传）**
   - 选择"指定路径"单选按钮
   - 输入插件文件路径：
     ```
     plugins/flow_matching_v2_plugin.py
     ```
     或绝对路径：
     ```
     /home/user/桌面/flow_matching_RD/webui/backend/plugins/flow_matching_v2_plugin.py
     ```

   **方式2：上传文件**
   - 选择"上传文件"单选按钮
   - 拖拽或选择文件：`webui/backend/plugins/flow_matching_v2_plugin.py`

4. **填写配置信息**：
   ```
   插件类名: FlowMatchingV2Plugin
   插件注册名: flow_matching_v2
   模型路径: /home/user/桌面/flow_matching_RD/outputs_v2/checkpoints/best_model.pth
   设备: cuda:0
   ```

5. **配置自定义参数（重要！）**
   - 点击"配置自定义参数 (JSON)"
   - 输入以下JSON：
   ```json
   {
     "base_channels": 64,
     "channel_mult": [1, 2, 4, 8],
     "attention_levels": [],
     "image_size": [512, 512]
   }
   ```
   - 点击"保存"

6. **点击"注册插件"**
   - 等待几秒钟
   - 看到"插件注册成功！"提示

7. **切换到"已注册插件"标签页**
   - 确认插件已出现在列表中

---

### 方式B：上传自定义插件

如果你想使用自己的模型：

1. **下载插件模板**
   - 在"开发指南"标签页中
   - 点击"下载 plugin_template.py"

2. **开发插件**
   - 参考 `PLUGIN_GUIDE.md` 开发你的插件
   - 实现5个必需方法

3. **上传注册**
   - 同方式A，上传你的插件文件
   - 填写配置信息
   - 注册完成

---

## 🎯 **开始推理**

### 1. 加载模型

1. 在右上角的下拉框中选择插件（如 `flow_matching_v2`）
2. 点击"加载模型"按钮
3. 确认参数（使用注册时的配置）
4. 点击"确定"

等待几秒钟，看到"模型加载成功"提示即可。

### 2. 上传图片

- 拖拽图片到上传区域
- 或点击上传区域选择文件
- 支持格式: PNG, JPG, JPEG, BMP

### 3. 调整参数

- **ODE步数**: 10-100（默认50）
- **设备**: cuda:0 / cuda:1 / cpu

### 4. 开始推理

点击"开始推理"按钮，等待2-5秒。

### 5. 查看结果

推理完成后，右侧会并排展示：
- **左图**: 输入图（Sim）
- **右图**: 输出图（Real）

可以点击"下载"按钮保存结果。

---

## 🔄 **插件管理功能**

### 查看插件列表

在"已注册插件"标签页：
- 查看所有已注册的插件
- 查看加载状态
- 查看当前使用的插件

### 切换插件

如果注册了多个插件：
1. 点击某个插件的"切换"按钮
2. 该插件变为当前使用的插件
3. 需要重新加载模型

### 查看插件详情

点击"详情"按钮查看：
- 插件名称
- 加载状态
- 模型信息（名称、版本、参数量等）

### 删除插件

点击"删除"按钮可注销插件：
- ⚠️ 如果模型已加载，会自动卸载
- ⚠️ 删除后需要重新注册才能使用

---

## 💡 **路径模式 vs 上传模式**

在注册插件时，推荐使用**路径模式**：

| 特性 | 指定路径 ⭐ | 上传文件 |
|------|-----------|---------|
| 速度 | ✅ 即时（无上传） | ⚠️ 需要上传 |
| 大文件 | ✅ 无限制 | ❌ 限制50MB |
| 已有文件 | ✅ 直接使用 | ❌ 需要重新上传 |
| 适用场景 | 服务器上已有插件 | 从本地上传新插件 |

**路径模式支持**：
- ✅ 绝对路径：`/home/user/桌面/flow_matching_RD/webui/backend/plugins/xxx.py`
- ✅ 相对路径：`plugins/xxx.py`（相对于 `backend/` 目录）

---

## 💡 **动态注册 vs 代码注册**

| 特性 | 动态注册（当前） | 代码注册 |
|------|----------------|---------|
| 修改插件 | ✅ 无需重启服务 | ❌ 需要重启服务 |
| 易用性 | ✅ 前端UI操作 | ⚠️ 需要编辑代码 |
| 多用户 | ✅ 支持 | ❌ 不适合 |
| 启动速度 | ✅ 快（按需加载） | ⚠️ 启动时注册 |
| 适用场景 | 生产环境、多用户 | 开发环境、单一模型 |

---

## 📋 **快速配置示例**

### Flow Matching V2 插件配置（复制粘贴即用）

```
【插件文件】
方式: 指定路径
路径: plugins/flow_matching_v2_plugin.py

【基础配置】
插件类名: FlowMatchingV2Plugin
插件注册名: flow_matching_v2
模型路径: /home/user/桌面/flow_matching_RD/outputs_v2/checkpoints/best_model.pth
设备: cuda:0

【自定义参数 (JSON)】
{
  "base_channels": 64,
  "channel_mult": [1, 2, 4, 8],
  "attention_levels": [],
  "image_size": [512, 512]
}
```

---

## 🔧 **进阶配置**

### 1. 同时支持两种方式

你可以在 `main.py` 中取消注释代码注册部分：

```python
# 在 startup_event 中
from plugins.flow_matching_v2_plugin import FlowMatchingV2Plugin

model_manager.register_plugin(
    plugin_name='flow_matching_v2',
    plugin_class=FlowMatchingV2Plugin,
    config={...}
)
```

这样启动时会自动注册该插件，同时仍可通过UI添加其他插件。

### 2. 持久化插件配置

创建 `plugins_config.json`，启动时自动加载：

```json
[
  {
    "plugin_name": "flow_matching_v2",
    "plugin_file": "plugins/flow_matching_v2_plugin.py",
    "plugin_class_name": "FlowMatchingV2Plugin",
    "config": {
      "checkpoint_path": "/path/to/model.pth",
      "device": "cuda:0",
      "base_channels": 64,
      "channel_mult": [1, 2, 4, 8],
      "attention_levels": [],
      "image_size": [512, 512]
    }
  }
]
```

在 `main.py` 中添加加载逻辑。

---

## 🐛 **常见问题**

### Q: 上传插件失败？
**A**: 
1. 确认文件是 `.py` 格式
2. 确认文件 < 50MB
3. 查看后端日志错误信息

### Q: 注册失败：类不存在？
**A**: 
1. 确认插件类名拼写正确（区分大小写）
2. 确认插件继承自 `InferenceInterface`
3. 查看后端日志详细错误

### Q: 注册失败：配置验证失败？
**A**: 
1. 确认 `checkpoint_path` 和 `device` 已填写
2. 如果插件需要自定义参数，确认已添加

### Q: 加载模型失败？
**A**: 
1. 确认模型文件路径正确且存在
2. 确认设备可用（GPU是否可用）
3. 查看后端日志错误信息

### Q: 如何更新插件代码？
**A**: 
1. 先删除旧插件
2. 修改插件文件
3. 重新上传注册

---

## 🎉 **完成！**

现在你可以：
- ✅ 通过UI动态添加/删除插件
- ✅ 无需重启服务即可更换模型
- ✅ 支持多个模型共存

详细开发文档：
- 完整指南: [`README.md`](./README.md)
- 插件开发: [`PLUGIN_GUIDE.md`](./PLUGIN_GUIDE.md)
- 生产部署: [`DEPLOYMENT.md`](./DEPLOYMENT.md)

---

**祝使用愉快！🚀**

