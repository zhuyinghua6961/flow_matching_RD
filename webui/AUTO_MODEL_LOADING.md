# 自动模型加载功能使用指南

## 📋 功能概述

WebUI现在支持自动扫描和加载trained_models目录下的所有训练好的模型，无需手动上传插件文件。

## 🎯 核心特性

- ✅ 自动扫描trained_models目录
- ✅ 从config.yaml自动提取训练参数
- ✅ 下拉列表选择模型
- ✅ 一键加载模型
- ✅ 显示模型详细信息（epoch, val_loss等）

## 📁 目录结构约定

为了让系统能够自动识别和加载模型，请按照以下约定组织目录结构：

```
项目根目录/
├── config_v2.yaml                 # ✅ 全局配置文件（所有模型共用）
├── train_v2.py
├── models_v2/
└── trained_models/                # 训练好的模型存放目录
    ├── project1/                  # 项目目录（可自定义名称）
    │   └── checkpoints/           # 模型检查点目录（必需）
    │       ├── best_model.pth
    │       ├── final_model.pth
    │       └── checkpoint_epoch_100.pth
    │
    ├── history_models/            # 历史模型分类（可嵌套）
    │   └── project2/
    │       └── checkpoints/
    │           └── model.pth
    │
    └── outputs_latest/            # 最新训练输出
        └── checkpoints/
            └── best_model.pth
```

### 必需文件

1. **`config_v2.yaml`** (或 `config.yaml`)
   - **位置**：项目根目录（与train_v2.py同级）
   - **用途**：提供normalize_mean/std、模型架构等参数
   - **说明**：所有模型共用这一个配置文件，无需在每个模型目录复制
   - **可选**：如果某个模型需要特殊配置，可以在其目录下放置单独的config_v2.yaml（会覆盖全局配置）

2. **`checkpoints/`** 目录
   - 包含所有的.pth模型文件
   - 每个.pth文件都会被扫描并添加到下拉列表

### 核心改进 ✨

- ✅ **不需要复制config文件** - 只需项目根目录有一个config_v2.yaml
- ✅ **简化目录结构** - 直接复制checkpoints目录即可
- ✅ **可选本地覆盖** - 需要特殊配置时才在模型目录添加config

## 🚀 使用流程

### 1. 准备模型

训练完成后，将checkpoints目录复制到trained_models：

```bash
# 方法1：只复制checkpoints目录（推荐）
cp -r outputs_v2_freq_ultra/checkpoints trained_models/project_1102/

# 方法2：复制整个输出目录
cp -r outputs_v2_freq_ultra trained_models/

# 方法3：复制到特定分类
mkdir -p trained_models/history_models/project_1102
cp -r outputs_v2_freq_ultra/checkpoints trained_models/history_models/project_1102/

# 方法4：创建软链接（不占用额外空间）
ln -s $(pwd)/outputs_v2_freq_ultra/checkpoints trained_models/latest/checkpoints
```

**说明**：
- ✅ 只需要复制checkpoints目录
- ✅ config_v2.yaml从项目根目录自动读取
- ✅ 大大简化了操作流程

### 2. 启动WebUI

```bash
cd /home/user/桌面/flow_matching_RD/webui

# 启动后端
cd backend
python main.py

# 启动前端（新终端）
cd ../frontend
npm run dev
```

### 3. 使用界面

1. **自动扫描**
   - WebUI启动时会自动扫描trained_models目录
   - 也可以点击"扫描模型"按钮手动刷新

2. **选择模型**
   - 从下拉列表中选择模型
   - 模型按项目分组显示
   - 显示epoch、val_loss、是否有配置等信息

3. **加载模型**
   - 点击"加载模型"按钮
   - 系统自动：
     - 从config.yaml提取参数
     - 注册插件
     - 加载模型
   - 加载成功后即可进行推理

## 🔧 配置参数说明

系统会从config_v2.yaml中自动提取以下参数：

### 数据配置（data）
```yaml
data:
  normalize_mean: 0.35  # ⚠️ 重要！必须与训练时一致
  normalize_std: 0.06   # ⚠️ 重要！必须与训练时一致
```

### 推理配置（inference）
```yaml
inference:
  ode_steps: 50         # ODE求解步数
  ode_method: "euler"   # ODE求解方法
```

### 模型配置（model）
```yaml
model:
  base_channels: 64
  channel_mult: [1, 2, 4, 8]
  attention_levels: []
  dropout: 0.1
```

## 📊 模型信息展示

下拉列表中每个模型会显示：

- **模型名称**：文件名（如best_model.pth）
- **Epoch**：训练轮数（从checkpoint读取）
- **Val Loss**：验证损失（从checkpoint读取）
- **配置状态**：是否有config.yaml

选择模型后，详细信息面板会显示：
- 模型ID
- 项目名称
- 文件路径
- 完整的训练参数

## 🎯 最佳实践

### 1. 命名规范

建议使用有意义的项目名称：

```
trained_models/
├── v2_freq_ultra_1102/      ✅ 好：包含版本和日期
├── baseline_50epochs/        ✅ 好：描述性强
├── outputs1/                 ❌ 差：不够明确
└── temp/                     ❌ 差：无意义
```

### 2. 使用全局配置

✅ **正确做法**（推荐）：
```bash
# 只需复制checkpoints目录
mkdir -p trained_models/project1
cp -r outputs_v2/checkpoints trained_models/project1/

# config_v2.yaml已经在项目根目录，无需复制
```

⚠️ **特殊情况**：
```bash
# 如果某个模型使用了不同的训练参数，可以保留其config
cp -r outputs_v2 trained_models/project1/  # 保留完整目录
```

### 3. 管理历史模型

```
trained_models/
├── production/              # 生产环境模型
│   └── v2_best/
├── experiments/             # 实验模型
│   ├── exp_lr_0001/
│   └── exp_augment/
└── archive/                 # 归档模型
    └── old_versions/
```

### 4. 定期清理

```bash
# 删除不需要的checkpoint
cd trained_models/project1/checkpoints
rm checkpoint_epoch_*.pth  # 保留best和final即可

# 归档旧模型
mv old_project archive/
```

## 🐛 常见问题

### Q1: 扫描不到模型？

**检查**：
1. 目录结构是否正确（必须有checkpoints/子目录）
2. 路径是否正确（trained_models在项目根目录）
3. 文件权限是否正确

```bash
# 检查目录结构
tree trained_models/

# 检查权限
ls -la trained_models/
```

### Q2: 加载失败提示"无配置文件"？

**原因**：项目根目录缺少config_v2.yaml

**解决**：
```bash
# 确保项目根目录有config_v2.yaml
ls config_v2.yaml  # 检查是否存在

# 如果不存在，从训练输出复制
cp outputs_v2_freq_ultra/config_v2.yaml ./
```

**说明**：
- config_v2.yaml应该在项目根目录（与train_v2.py同级）
- 不需要在每个trained_models子目录复制

### Q3: 推理效果不对？

**原因**：normalize参数不匹配

**检查**：
```yaml
# config_v2.yaml中的normalize参数
data:
  normalize_mean: 0.35  # 必须与训练时一致
  normalize_std: 0.06
```

### Q4: 模型太多，加载很慢？

**优化**：
1. 只保留best_model.pth和final_model.pth
2. 将不常用的模型移到archive目录
3. 使用软链接而不是复制

```bash
# 清理checkpoint
cd checkpoints/
ls | grep checkpoint_epoch | xargs rm
```

## 📝 API接口

如果需要通过API调用：

### 扫描模型
```bash
curl http://localhost:8000/api/models/scan
```

### 获取已扫描的模型
```bash
curl http://localhost:8000/api/models/scanned
```

### 自动加载模型
```bash
curl -X POST http://localhost:8000/api/models/auto_load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "project1/best_model.pth", "device": "cuda:0"}'
```

## 📚 示例

### 完整示例

```bash
# 1. 训练模型
python train_v2.py --config config_v2.yaml

# 2. 复制到trained_models
cp -r outputs_v2_freq_ultra trained_models/experiment_1102

# 3. 启动WebUI
cd webui/backend && python main.py

# 4. 在浏览器中：
# - 打开 http://localhost:8000
# - 点击"扫描模型"
# - 从下拉列表选择 "experiment_1102/best_model.pth"
# - 点击"加载模型"
# - 上传图像并推理
```

## ⚙️ 高级配置

### 自定义扫描目录

修改`webui/backend/main.py`：

```python
# 启动时扫描
models = model_manager.scan_trained_models(base_dir="custom_models")
```

### 使用不同的配置文件名

系统会按顺序查找：
1. `config_v2.yaml`
2. `config.yaml`
3. `config_v2.yml`
4. `config.yml`

## 🎉 总结

通过这个功能，你可以：
- ✅ 无需手动上传插件文件
- ✅ 快速切换不同训练的模型
- ✅ 自动提取正确的训练参数
- ✅ 管理多个模型版本

享受更便捷的推理体验！🚀

