# 训练指南

完整的训练、推理和参数配置说明。

---

## 📦 数据准备

### 数据目录结构

**每个图片对应一个独立的prompt文件：**

```
data/
├── sim/              # 仿真RD图
│   ├── rd001.png
│   ├── rd002.png
│   └── ...
├── real/             # 真实RD图
│   ├── rd001.png
│   ├── rd002.png
│   └── ...
└── prompt/           # 提示词文件（每个图片对应一个txt）
    ├── rd001.txt
    ├── rd002.txt
    └── ...
```

**配对规则**：同名文件构成一组数据（如 `rd001.png` + `rd001.txt`）

### Prompt文件格式

**单目标** (`rd001.txt`):
```
radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 1, the first target: distance = 102m, velocity = 20.00m/s.
```

**双目标** (`rd002.txt`):
```
radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s.
```

**三目标** (`rd003.txt`):
```
radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 3, the first target: distance = 79m, velocity = -27.00m/s, the second target: distance = 126m, velocity = -18.00m/s, the third target: distance = 26m, velocity = 26.00m/s.
```

### 数据要求

- ✅ 图像格式：PNG
- ✅ 分辨率：任意（自动resize到512×512）
- ✅ 配对关系：sim、real、prompt三个文件必须同名
- ✅ 数量建议：至少500对
- ✅ Prompt文件：UTF-8编码的TXT文件

### 数据验证

```bash
python utils/dataset.py
```

---

## 🚀 训练

### 基础训练

```bash
# 使用默认配置
python train.py

# 指定数据目录
python train.py --data_root ./data

# 调整batch size
python train.py --batch_size 8

# 调整学习率
python train.py --lr 5e-5
```

### 从检查点恢复

```bash
# 从最新检查点
python train.py --resume ./checkpoints/latest.pth

# 从最佳模型
python train.py --resume ./checkpoints/best_model.pth

# 从特定epoch
python train.py --resume ./checkpoints/checkpoint_epoch_0050.pth
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置文件路径 | `--config my_config.yaml` |
| `--data_root` | 数据根目录 | `--data_root ./data` |
| `--batch_size` | 批大小 | `--batch_size 8` |
| `--num_epochs` | 训练轮数 | `--num_epochs 200` |
| `--lr` | 学习率 | `--lr 5e-5` |
| `--resume` | 恢复检查点路径 | `--resume ./checkpoints/latest.pth` |
| `--device` | 设备 | `--device cuda` |

---

## ⚙️ 配置文件

所有参数在 `config.yaml` 中配置：

### 数据配置

```yaml
data:
  data_root: "./data"
  img_size: 512
  max_speed: 30.0        # RD图最大速度范围 ±30m/s
  max_range: 200.0       # RD图最大距离 0-200m
```

### 模型配置

```yaml
model:
  unet:
    base_channels: 64
    channel_mult: [1, 2, 4, 8]
  controlnet:
    base_channels: 32
    channel_mult: [1, 2, 4, 8]
```

### Loss配置

```yaml
loss:
  weight_factor: 50      # 目标区域权重 (30-100)
  threshold: 0.1         # 热力图阈值
  focal_gamma: 0.0       # Focal Loss gamma
```

### 训练配置

```yaml
train:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001
  
  # 检查点保存
  save_interval: 5       # 每5个epoch保存
  save_best_only: false
  keep_last_n_checkpoints: 5
  
  # 早停机制
  early_stopping:
    enabled: true
    patience: 20         # 20个epoch无改善则停止
    min_delta: 0.0001
    monitor: "loss"
```

---

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir ./logs --port 6006
```

浏览器访问：`http://localhost:6006`

### 关键指标

- `train/loss`: 总损失
- `train/best_loss`: 历史最佳损失
- `train/target_loss`: 目标区域损失
- `train/bg_loss`: 背景损失
- `train/target_ratio`: 目标占比

---

## 🔮 推理

### 单张图片推理

```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pth \
  --sim_rd ./test_images/rd001.png \
  --prompt "radar-RD-map; ... target number = 1, ..." \
  --output ./results/real_rd001.png
```

### 可视化结果

```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pth \
  --sim_rd ./test.png \
  --prompt "速度: 5m/s, 距离: 100m" \
  --output ./result.png \
  --visualize
```

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必需 | 模型检查点路径 |
| `--sim_rd` | 必需 | 仿真RD图路径 |
| `--prompt` | 必需 | 文本描述 |
| `--output` | `./output.png` | 输出路径 |
| `--steps` | `20` | ODE求解步数 |
| `--method` | `euler` | ODE方法 (euler/rk4) |
| `--visualize` | `False` | 是否可视化 |

---

## 🎛️ 参数调优

### 模型参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `unet_base_channels` | 64 | 显存小→32，追求质量→96 |
| `controlnet_base_channels` | 32 | 通常为UNet的一半 |
| `channel_mult` | (1,2,4,8) | 轻量化→(1,2,4) |
| `num_res_blocks` | 2 | 不建议超过3 |
| `attention_levels` | (2,3) | 仅在低分辨率 |

### Loss参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `weight_factor` | 50 | 30-100，过大可能过拟合 |
| `loss_threshold` | 0.1 | 根据sigma调整 |
| `focal_gamma` | 0.0 | 难样本多→2.0 |
| `use_perceptual` | False | 细节要求高→True |

### 热力图参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `heatmap_sigma` | 10.0 | 目标大→增大，小→减小 |
| `max_speed` | 30.0 | 根据实际RD图调整 |
| `max_range` | 200.0 | 根据实际RD图调整 |

### 训练参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `batch_size` | 4 | 显存允许尽量大 |
| `gradient_accumulation_steps` | 1 | 显存不足时增大，实际batch=batch_size×steps |
| `learning_rate` | 1e-4 | 不稳定→5e-5 |
| `num_epochs` | 100 | 有早停可设大 |

### 检查点参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `save_interval` | 5 | 频繁保存→3，节省空间→10 |
| `save_best_only` | False | 空间紧张→True |
| `keep_last_n_checkpoints` | 5 | 0=全部保留，3-10合适 |

### 早停参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `early_stop_patience` | 20 | 数据少→10，数据多→30 |
| `early_stop_min_delta` | 0.0001 | 根据loss量级调整 |
| `early_stop_monitor` | "loss" | 或"target_loss" |

---

## 🔧 常见问题

### Q1: 显存不足

**方法1: 使用梯度累积**（推荐）
```yaml
train:
  batch_size: 2              # 减小batch size
  gradient_accumulation_steps: 4  # 增加累积步数
  # 实际等效batch size = 2 × 4 = 8
```

**方法2: 减小batch size**
```bash
python train.py --batch_size 2
```

**方法3: 禁用混合精度**（不推荐）
```yaml
train:
  mixed_precision: false
```

### Q2: 训练不收敛

```bash
# 降低学习率
python train.py --lr 5e-5
```

或检查：
- Loss是否正常（target_loss应该远大于bg_loss）
- 数据是否配对正确
- 热力图是否准确：`python utils/heatmap.py`

### Q3: 生成结果模糊

```bash
# 增加ODE步数
python inference.py --steps 50

# 或使用RK4方法
python inference.py --method rk4
```

或修改 `config.yaml`:
```yaml
loss:
  use_perceptual: true
  weight_factor: 80
```

### Q4: 训练过早停止

修改 `config.yaml`:
```yaml
train:
  early_stopping:
    enabled: false    # 禁用早停
    # 或增加容忍度
    patience: 50
```

---

## 📈 训练时间估算

| 配置 | Batch Size | 数据量 | 每Epoch | 100 Epochs |
|------|------------|--------|---------|------------|
| RTX 3090 | 4 | 1000对 | ~5分钟 | ~8小时 |
| RTX 3090 | 8 | 1000对 | ~3分钟 | ~5小时 |
| RTX 4090 | 8 | 1000对 | ~2分钟 | ~3.5小时 |

**提示**：启用早停机制后，实际训练时间通常会减少30-50%

---

## 🎓 最佳实践

1. **从小规模验证**：先用100对数据快速验证流程
2. **监控指标**：重点关注 `target_loss` 和 `bg_loss` 的比值
3. **早停机制**：建议启用，节省时间
4. **定期保存**：`save_interval` 设置为5-10
5. **批量大小**：尽量使用8或更大（如果显存允许）
6. **学习率**：从1e-4开始，不稳定则降低
7. **验证数据**：训练前运行 `python utils/dataset.py`

---

**更新日期**: 2025-10-29

