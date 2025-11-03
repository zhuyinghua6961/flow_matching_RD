# 自动模型加载 - 快速开始

## 🚀 5分钟快速上手

### 步骤1：准备模型目录

```bash
cd /home/user/桌面/flow_matching_RD

# 方法1：只复制checkpoints（推荐，更简洁）
mkdir -p trained_models/my_model
cp -r outputs_v2_freq_ultra/checkpoints trained_models/my_model/

# 方法2：复制整个输出目录
cp -r outputs_v2_freq_ultra trained_models/

# 方法3：从历史输出复制
mkdir -p trained_models/history_models/experiment_old
cp -r outputs_v2_new_loss/checkpoints trained_models/history_models/experiment_old/

# 注意：config_v2.yaml在项目根目录，无需复制！
```

### 步骤2：启动WebUI

```bash
cd webui

# 终端1：启动后端
cd backend
python main.py
# 看到：发现 X 个模型

# 终端2：启动前端
cd frontend
npm run dev
# 浏览器打开 http://localhost:5173
```

### 步骤3：使用界面

1. **打开浏览器** → http://localhost:5173
2. **查看左侧"模型选择"卡片**
3. **从下拉列表选择模型** （例如：outputs_v2_freq_ultra/best_model.pth）
4. **点击"加载模型"按钮**
5. **等待加载完成** → 出现"加载成功"提示
6. **上传图像并开始推理** ✅

## 📁 目录结构示例

成功后，你的目录应该看起来像这样：

```
/home/user/桌面/flow_matching_RD/
├── config_v2.yaml                     ← ✅ 全局配置（必须有！）
├── train_v2.py
├── models_v2/
├── trained_models/                    ← 新建这个目录
│   ├── my_model/                      ← 只需要checkpoints目录
│   │   └── checkpoints/
│   │       ├── best_model.pth
│   │       └── final_model.pth
│   └── history_models/
│       └── experiment_old/
│           └── checkpoints/
│               └── best_model.pth
└── webui/
    ├── backend/
    └── frontend/
```

**核心改进**：
- ✅ config_v2.yaml在根目录，所有模型共用
- ✅ trained_models下只需要checkpoints目录
- ✅ 简化了目录结构，不需要重复复制配置文件

## ✅ 检查清单

在开始之前，确保：

- [ ] **项目根目录**有config_v2.yaml文件（与train_v2.py同级）
- [ ] trained_models目录已创建
- [ ] 至少有一个项目目录包含checkpoints/子目录
- [ ] checkpoints/目录下有.pth文件
- [ ] 后端服务器正在运行
- [ ] 前端开发服务器正在运行

**最重要**：确保根目录有config_v2.yaml！

## 🐛 遇到问题？

### 问题1：扫描不到模型

```bash
# 检查目录结构
tree trained_models/ -L 3

# 应该看到：
# trained_models/
# └── project_name/
#     ├── config_v2.yaml
#     └── checkpoints/
#         └── *.pth
```

### 问题2：加载失败

查看后端日志：
```bash
# 后端终端会显示详细错误信息
# 常见原因：
# - 缺少config_v2.yaml
# - normalize参数不对
# - GPU内存不足
```

### 问题3：提示"未找到配置文件"

```bash
# 检查根目录是否有config_v2.yaml
cd /home/user/桌面/flow_matching_RD
ls config_v2.yaml

# 如果不存在，从训练输出复制
cp outputs_v2_xxx/config_v2.yaml ./

# 或者检查是否在webui目录里启动了后端（错误）
# 应该确保config_v2.yaml在项目根目录
```

## 💡 提示

1. **全局配置**：config_v2.yaml放在项目根目录，所有模型共用
2. **首次使用**：点击"扫描模型"按钮刷新列表
3. **切换模型**：直接从下拉列表选择另一个模型并加载
4. **查看详情**：选择模型后可以看到epoch、loss等信息
5. **简化操作**：只需复制checkpoints目录，无需复制config文件

## 📚 完整文档

详细说明请查看：
- [AUTO_MODEL_LOADING.md](./AUTO_MODEL_LOADING.md) - 完整功能文档
- [QUICKSTART_DYNAMIC.md](./QUICKSTART_DYNAMIC.md) - 动态插件加载文档

## 🎉 成功！

如果你看到：
- ✅ 模型列表显示你的模型
- ✅ 点击加载后显示"加载成功"
- ✅ 可以正常推理

恭喜！你已经成功使用自动模型加载功能！🚀
