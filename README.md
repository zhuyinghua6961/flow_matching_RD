# Flow Matching RD图 Sim2Real 映射

基于 **Flow Matching + ControlNet + UNet** 的轻量级无人机雷达RD图仿真到真实域映射模型。

---

## 🎯 项目简介

将仿真生成的无人机雷达RD图转换为真实域RD图。

**核心特点**：
- ✅ **轻量化设计**：总参数量 ~25M，适合单卡训练
- ✅ **精确位置控制**：基于热力图的ControlNet引导
- ✅ **Flow Matching**：训练稳定，推理高效（20步）
- ✅ **针对性Loss**：加权损失专为RD图稀疏目标优化

---

## 🏗️ 模型架构

### 整体流程

```
训练时：
sim_RD + real_RD + prompt
    ↓
Prompt → 热力图生成 (规则提取，无需训练)
    ↓
[ControlNet] sim_RD + Heatmap → 条件特征
    ↓ (注入)
[UNet] Flow Matching速度场预测
    ↓
加权Loss优化

推理时：
sim_RD + prompt → ODE求解 → real_RD
```

### 1. UNet主干

**参数配置**：
- 基础通道数: 64
- 通道倍数: (1, 2, 4, 8) → [64, 128, 256, 512]
- 下采样层数: 3层
- ResBlock数: 每层2个
- 注意力机制: 仅在低分辨率层 (64×64, 32×32)
- 总参数: ~18M

**特点**：
- 轻量化设计（相比标准UNet减少40%参数）
- 跳跃连接保留空间细节
- Self-Attention增强全局感知

### 2. ControlNet分支

**参数配置**：
- 基础通道数: 32（UNet的一半）
- 镜像UNet编码器结构
- Zero-Convolution初始化（训练稳定）
- 总参数: ~7M

**输入**：`concat[sim_RD(3通道), heatmap(1通道)]` → 4通道

**输出**：多尺度特征图注入UNet对应层

### 3. 热力图生成器

**完全规则化，无训练参数**：

1. **Prompt解析** → 提取速度、距离参数
2. **物理坐标 → 像素坐标**：
   - X轴：速度 → 横向像素 (中心=0m/s)
   - Y轴：距离 → 纵向像素 (顶部=0m)
3. **高斯热力图**：σ=10，归一化到[0,1]
4. **多目标**：多个高斯峰取最大值叠加

**支持格式**：
- 单/双/三目标：`radar-RD-map; ... target number = 2, ...`
- 简单格式：`速度: 5m/s, 距离: 100m`

---

## 📐 损失函数

### 加权Flow Matching Loss

**标准Flow Matching**：
```
L_fm = ||v_θ(x_t, t) - (x_1 - x_0)||²
```
其中：
- x_0 = sim_RD (源域)
- x_1 = real_RD (目标域)
- x_t = (1-t)·x_0 + t·x_1 + σ·ε (插值路径)
- v_θ = 模型预测的速度场

**加权策略**：

```python
# 从热力图生成权重图
weight_map = MaxPool(heatmap) > threshold
weight_map = weight_map * weight_factor + 1.0

# 加权Loss
loss = (weight_map * (pred - target)²).mean()
```

**关键参数**：
- `weight_factor`: 目标区域权重（默认50）
- `threshold`: 热力图阈值（默认0.1）
- `MaxPool`: 保留峰值区域

**为什么需要加权**：
- RD图中目标占比 < 1%
- 标准MSE会被背景主导
- 加权Loss强制模型关注目标区域

**效果对比**：
```
标准Loss: 背景精细，目标模糊
加权Loss: 目标清晰，背景稍差（可接受）
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

**主要依赖**：
- PyTorch >= 2.0
- torchvision
- numpy
- Pillow
- PyYAML

### 2. 数据准备

```
data/
├── sim/              # 仿真RD图
│   ├── rd001.png
│   └── ...
├── real/             # 真实RD图
│   ├── rd001.png
│   └── ...
└── prompt/           # 提示词（每个图片一个txt）
    ├── rd001.txt
    └── ...
```

**Prompt示例** (`rd001.txt`):
```
radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 1, the first target: distance = 102m, velocity = 20.00m/s.
```

### 3. 训练

```bash
python train.py --data_root ./data
```

### 4. 推理

```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pth \
  --sim_rd ./test.png \
  --prompt "速度: 5m/s, 距离: 100m" \
  --output ./result.png
```

---

## 📚 详细文档

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - 完整训练指南（数据准备、参数配置、命令行用法）
- **[config.yaml](config.yaml)** - 配置文件说明

---

## 📊 项目结构

```
flow_matching_RD/
├── models/
│   ├── unet.py              # UNet主干
│   ├── controlnet.py        # ControlNet分支
│   └── flow_matching.py     # Flow Matching封装
├── utils/
│   ├── dataset.py           # 数据加载器
│   ├── heatmap.py           # 热力图生成器
│   ├── loss.py              # 加权损失函数
│   └── early_stopping.py    # 早停机制
├── config.yaml              # 配置文件
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
└── README.md
```

---

## 🎓 引用

本项目基于以下工作：
- Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- ControlNet: [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

---

**更新日期**: 2025-10-29
