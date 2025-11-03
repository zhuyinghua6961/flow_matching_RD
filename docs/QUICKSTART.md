# 快速开始指南

本文档提供Flow Matching RD图Sim2Real模型的完整使用流程。

---

## 📁 数据准备

### 目录结构

```
dataset/
├── train/
│   ├── sim/    # 仿真RD图（PNG灰度图，512×512）
│   └── real/   # 真实RD图（文件名需对应）
├── val/
│   ├── sim/
│   └── real/
└── test/
    ├── sim/
    └── real/
```

### 要求

- **格式**: PNG灰度图（单通道）
- **分辨率**: 512×512
- **命名**: sim和real通过文件名匹配（如`rd001.png`）
- **归一化**: 像素值[0,255] → [0,1] → `(x-0.35)/0.06`

### 数据量建议

- 最少: 200对
- 推荐: 500+对
- 验证集: 总数据的10-15%
- 测试集: 50-100对

---

## 🚀 训练

### 基础训练

```bash
python train_v2.py --config config_v2.yaml
```

### 关键参数（config_v2.yaml）

```yaml
# 数据路径
data:
  train_root: "./new_dataset/train"
  val_root: "./new_dataset/val"
  test_root: "./new_dataset/test"

# Loss权重（核心配置）
loss:
  use_frequency: true
  frequency_weight: 2.5      # 频域Loss权重（主导）
  use_ssim: true
  ssim_weight: 0.5           # SSIM Loss权重（辅助）
  use_perceptual: false      # 关闭VGG感知损失

# 训练配置
train:
  batch_size: 1
  gradient_accumulation_steps: 16  # 实际batch=16
  learning_rate: 0.00005
  num_epochs: 200
  lr_scheduler: "plateau"    # 自适应学习率
  early_stopping:
    enabled: true
    patience: 30             # 30轮无改善则停止
```

### 输出文件

训练过程会在`outputs_v2_freq_ultra/`目录下生成：

```
outputs_v2_freq_ultra/
├── checkpoints/
│   ├── best_model.pth          # 验证Loss最低的模型
│   ├── final_model.pth         # 最后一轮的模型
│   ├── checkpoint_epoch_*.pth  # 每10轮保存的检查点
├── logs/                       # TensorBoard日志
└── results/                    # 训练过程的可视化结果
```

### 训练监控

```bash
# 启动TensorBoard
tensorboard --logdir outputs_v2_freq_ultra/logs

# 访问 http://localhost:6006
```

**关键指标**:
- `train/loss_fm`: Flow Matching Loss
- `train/loss_frequency`: **频域Loss（重点观察）**
- `train/loss_ssim`: SSIM Loss
- `val/val_loss`: 验证总Loss（用于早停）

### 训练诊断

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 频域Loss不降 | 权重不够 | 增大`frequency_weight`到3.0 |
| 输出过于平滑 | SSIM权重过高 | 降低`ssim_weight`到0.3 |
| 训练不稳定 | 学习率过大 | 降低`learning_rate`到3e-5 |
| Loss为NaN | 梯度爆炸 | 检查归一化，降低学习率 |
| 收敛太慢 | 学习率过小 | 增大到1e-4 |

---

## 🔮 推理

### 方式1：命令行推理

#### 单张图像

```bash
python inference_v2.py \
    --checkpoint outputs_v2_freq_ultra/checkpoints/best_model.pth \
    --input test.png \
    --output result.png \
    --ode_steps 50
```

#### 批量推理

```bash
python inference_v2.py \
    --checkpoint outputs_v2_freq_ultra/checkpoints/best_model.pth \
    --input dataset/test/sim/ \
    --output outputs_v2_freq_ultra/results/ \
    --batch
```

**参数说明**:
- `--ode_steps`: ODE求解步数（30-100）
  - 30: 快速推理，质量略降
  - 50: 平衡速度和质量（推荐）
  - 100: 最高质量，速度慢
- `--ode_method`: 求解方法
  - `euler`: 快速（默认）
  - `rk4`: 精确但慢

### 方式2：WebUI推理（推荐）

#### 启动后端

```bash
cd webui/backend
python main.py

# 后端启动在 http://localhost:8000
```

#### 启动前端

```bash
cd webui/frontend
npm install  # 首次运行需要
npm run dev

# 前端启动在 http://localhost:5173
```

#### 使用WebUI

1. **自动加载模型**
   - 后端启动时自动扫描`trained_models/`目录
   - 前端点击"扫描模型"→选择模型→"加载模型"
   - 显存占用会增加到2.5-3GB（正常）

2. **上传图像**
   - 支持拖拽或点击上传
   - 支持PNG/JPG格式

3. **推理**
   - 调整ODE步数（可选）
   - 点击"开始推理"
   - 大约8秒生成结果

4. **查看结果**
   - 支持对比显示（输入vs输出）
   - 支持下载生成图像

---

## 📊 模型测试

### 评估指标

```bash
python test_v2.py \
    --checkpoint outputs_v2_freq_ultra/checkpoints/best_model.pth \
    --save_results \
    --output_dir outputs_v2_freq_ultra/test_results/
```

**输出指标**:
- **MSE**: 均方误差（越低越好）
- **PSNR**: 峰值信噪比（越高越好，通常>20dB）
- **SSIM**: 结构相似度（0-1，越接近1越好）
- **频域MSE**: 频谱匹配度（核心指标）

### 可视化对比

生成的结果会保存在`test_results/`目录，包含：
- 输入图像（sim）
- 生成图像（generated）
- 真实图像（real）
- 三者对比图

---

## ⚙️ 数据增强

**当前配置**（`config_v2.yaml`中设置`augment: true`）:

```python
数据增强策略:
- ✅ 垂直翻转（上下翻转）: 50%概率
- ✅ 亮度调整: ±10%
- ✅ 对比度调整: ±10%
- ❌ 水平翻转（左右翻转）: 禁用
```

**为什么不用水平翻转？**
- 雷达RD图有物理方向性：
  - 垂直轴 = 速度轴 → 翻转等价于正负速度对调 ✅
  - 水平轴 = 距离轴 → 翻转会改变目标距离关系 ❌

---

## 🔧 常见问题

### Q1: 生成的图像没有多普勒十字？

**A**: 增大频域Loss权重
```yaml
loss:
  frequency_weight: 3.0  # 从2.5增大到3.0
```

### Q2: 输出图像过于平滑，缺少细节？

**A**: 降低SSIM权重
```yaml
loss:
  ssim_weight: 0.3  # 从0.5降低到0.3
```

### Q3: 显存不足？

**A**: 调整batch配置
```yaml
train:
  batch_size: 1
  gradient_accumulation_steps: 32  # 保持总batch=32
```

或关闭attention:
```yaml
model:
  attention_levels: []  # 空列表表示不使用attention
```

### Q4: WebUI推理返回404？

**A**: 检查：
1. 后端是否正常启动（端口8000）
2. 前端是否指向正确的后端地址
3. 模型是否成功加载（查看后端日志）

### Q5: 训练速度太慢？

**A**: 
1. 减少ODE步数验证:
   ```yaml
   inference:
     ode_steps: 30  # 训练中的验证可以用少一点
   ```
2. 减少验证频率:
   ```yaml
   train:
     val_interval: 5  # 每5轮验证一次
   ```

---

## 📈 调参优先级

按重要性排序：

1. **频域Loss权重** (`frequency_weight`): 2.0-3.0
   - 直接影响多普勒学习效果
   
2. **学习率** (`learning_rate`): 3e-5 ~ 1e-4
   - 影响收敛速度和稳定性
   
3. **SSIM权重** (`ssim_weight`): 0.3-0.8
   - 平衡结构保持和细节丰富度
   
4. **早停patience** (`early_stopping.patience`): 20-40
   - 小数据集用大patience，避免过早停止
   
5. **梯度累积** (`gradient_accumulation_steps`): 8-32
   - 显存小用大值，保证总batch足够大

---

## 🎯 推荐工作流

### 第一次训练（快速验证）

```yaml
train:
  num_epochs: 50
  learning_rate: 0.0001
loss:
  frequency_weight: 2.0
  ssim_weight: 0.5
```

**目的**: 快速验证数据和模型是否正常工作

### 正式训练（追求效果）

```yaml
train:
  num_epochs: 200
  learning_rate: 0.00005
loss:
  frequency_weight: 2.5
  ssim_weight: 0.5
data:
  augment: true
```

**目的**: 完整训练，获得最佳效果

### 极致优化（竞赛/论文）

```yaml
train:
  num_epochs: 300
  learning_rate: 0.00003
loss:
  frequency_weight: 3.0
  ssim_weight: 0.3
data:
  augment: true
```

**目的**: 榨取模型极限性能

---

## 📦 模型部署

### 将训练好的模型放入WebUI

```bash
# 创建目录
mkdir -p trained_models/my_model/checkpoints

# 复制模型文件
cp outputs_v2_freq_ultra/checkpoints/best_model.pth \
   trained_models/my_model/checkpoints/

# 启动WebUI后会自动扫描并加载
```

### Python API调用

```python
import torch
from models_v2 import Sim2RealFlowModel
from PIL import Image
from torchvision import transforms

# 加载模型
model = Sim2RealFlowModel()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# 预处理
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.35], std=[0.06])
])

# 推理
sim_img = Image.open('sim.png').convert('L')
sim_tensor = transform(sim_img).unsqueeze(0).cuda()

with torch.no_grad():
    real_tensor = model.inference(sim_tensor, ode_steps=50)

# 后处理
real_tensor = real_tensor * 0.06 + 0.35
real_img = transforms.ToPILImage()(real_tensor[0])
real_img.save('real.png')
```

---

## 🔄 版本历史

- **V2 (Current)**: 频域Loss主导，端到端，无需prompt
- **V1**: Perceptual Loss + ControlNet，需要prompt

**V2优势**: 
- 更快（推理速度+30%）
- 更准确（频域Loss直接约束物理特征）
- 更易用（无需准备prompt）

---

## 📧 技术支持

如遇到问题，请检查：
1. 配置文件格式是否正确
2. 数据路径是否存在
3. 训练日志中的错误信息
4. GPU显存是否充足

关键日志位置:
- 训练日志: `outputs_v2_freq_ultra/logs/`
- 后端日志: 终端输出或`webui/backend/backend.log`

