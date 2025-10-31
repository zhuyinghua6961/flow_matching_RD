# V2模型快速入门指南

> 纯图像对的Sim2Real模型 - 无需prompt！

---

## ⚡ 5分钟快速开始

### **1. 准备数据**

```bash
# 数据目录结构（简化了！不需要prompt/）
dataset/
├── train/
│   ├── sim/     # 仿真图
│   └── real/    # 真实图
├── val/
│   ├── sim/
│   └── real/
└── test/
    ├── sim/
    └── real/
```

**要求**：
- ✅ sim和real通过文件名匹配（如rd001.png）
- ✅ PNG灰度图
- ✅ 至少1000对训练数据

---

### **2. 训练模型**

```bash
# 直接开始训练！
python train_v2.py --config config_v2.yaml
```

**监控训练**：
```bash
# 新开一个终端
tensorboard --logdir outputs_v2/logs
```

访问 http://localhost:6006 查看训练曲线

---

### **3. 推理生成**

```bash
# 单张推理
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input dataset/test/sim/rd001.png \
    --output result.png

# 批量推理
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input dataset/test/sim/ \
    --output outputs_v2/results/ \
    --batch
```

**就这么简单！不需要提供prompt！**

---

### **4. 评估模型**

```bash
python test_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --save_results \
    --output_dir outputs_v2/test_results/
```

输出指标：MSE, PSNR, SSIM

---

## 🔧 常见问题

### **Q1: 训练出现NaN怎么办？**

修改`config_v2.yaml`：
```yaml
loss:
  perceptual_weight: 0.01  # 降低到0.01

train:
  mixed_precision: false   # 改为false
  learning_rate: 0.00005   # 降低学习率
```

---

### **Q2: 显存不足怎么办？**

```yaml
train:
  batch_size: 2            # 降低batch size
  gradient_accumulation_steps: 8  # 增加梯度累积

model:
  attention_levels: []     # 关闭attention
  base_channels: 32        # 降低通道数
```

---

### **Q3: 生成图像太模糊？**

```yaml
loss:
  perceptual_weight: 0.03  # 增大感知损失权重（0.02-0.05）

inference:
  ode_steps: 100           # 增加ODE步数
  ode_method: "rk4"        # 使用更精确的方法
```

---

### **Q4: 训练太慢？**

```yaml
train:
  mixed_precision: true    # 开启混合精度（稳定后）
  
loss:
  perceptual_interval: 20  # 降低perceptual loss计算频率
  
inference:
  ode_steps: 30            # 推理时用少步（快速）
```

---

## 📊 参数调整建议

### **基础配置（快速验证）**
```yaml
model:
  base_channels: 32
  
train:
  batch_size: 4
  num_epochs: 20
  
loss:
  perceptual_weight: 0.01
```
→ 20 epochs，看看能否学到映射

---

### **标准配置（推荐）**
```yaml
model:
  base_channels: 64
  
train:
  batch_size: 4
  num_epochs: 100
  gradient_accumulation_steps: 4
  
loss:
  perceptual_weight: 0.02
  
inference:
  ode_steps: 50
```
→ 平衡效果和速度

---

### **高质量配置（追求效果）**
```yaml
model:
  base_channels: 64
  
train:
  batch_size: 8
  num_epochs: 200
  
loss:
  perceptual_weight: 0.03
  
inference:
  ode_steps: 100
  ode_method: "rk4"
```
→ 最佳效果，但训练时间长

---

## 🎯 关键参数说明

| 参数 | 作用 | 推荐值 | 调整建议 |
|------|------|--------|---------|
| `perceptual_weight` | 感知损失权重 | 0.01-0.03 | **最关键！** 太大NaN，太小模糊 |
| `ode_steps` | 推理步数 | 50 | 30快，100精细 |
| `batch_size` | 批大小 | 4 | 根据显存调整 |
| `base_channels` | 模型大小 | 64 | 32快，64效果好 |
| `mixed_precision` | 混合精度 | false | 稳定后可改true加速 |

---

## 📈 训练监控

### **正常训练**
```
Epoch 1:  Train Loss: 0.52, Val Loss: 0.48
Epoch 10: Train Loss: 0.35, Val Loss: 0.34
Epoch 50: Train Loss: 0.22, Val Loss: 0.24
Epoch 100: Train Loss: 0.18, Val Loss: 0.20
```
→ Loss稳定下降，验证损失接近训练损失

---

### **过拟合**
```
Epoch 50: Train Loss: 0.12, Val Loss: 0.35
```
→ 训练损失远小于验证损失
→ 解决：增加数据，启用数据增强`augment: true`

---

### **欠拟合**
```
Epoch 100: Train Loss: 0.45, Val Loss: 0.46
```
→ 损失停留在高位
→ 解决：增大模型`base_channels: 128`，训练更久

---

## 🚀 优化流程

### **第一轮：验证可行性（1-2小时）**
```bash
# config_v2.yaml设置
num_epochs: 20
base_channels: 32
perceptual_weight: 0.01

# 训练
python train_v2.py

# 观察：
# - Loss是否下降？
# - 生成图是否有sim→real的变化？
```

---

### **第二轮：完整训练（1-2天）**
```bash
# 调整配置
num_epochs: 100
base_channels: 64
perceptual_weight: 0.02

# 训练
python train_v2.py

# 观察：
# - Val Loss最低点
# - 生成质量如何
```

---

### **第三轮：调优（根据需要）**

| 问题 | 解决方案 |
|------|---------|
| 太模糊 | 增大`perceptual_weight`到0.03-0.05 |
| 杂波太多/噪声大 | 降低`perceptual_weight`到0.01 |
| 过拟合 | 数据增强`augment: true` |
| 训练慢 | 混合精度`mixed_precision: true` |

---

## 📁 输出文件说明

```
outputs_v2/
├── checkpoints/
│   ├── best_model.pth           # 最佳模型
│   ├── checkpoint_epoch_10.pth  # 周期性检查点
│   └── ...
├── logs/
│   └── events.out.tfevents...   # TensorBoard日志
└── results/
    └── generated/               # 推理结果
```

---

## 💡 Pro Tips

1. **先用小模型快速验证**
   - `base_channels: 32`, `num_epochs: 20`
   - 确认数据没问题，模型能学

2. **perceptual_weight是灵魂**
   - 0.01: 安全，不会NaN
   - 0.02: 推荐，效果和稳定性平衡
   - 0.03-0.05: 追求效果，需要监控NaN

3. **观察TensorBoard**
   - `train/loss_fm`: Flow Matching Loss
   - `val/loss`: 验证损失（最重要）
   - `train/loss_perceptual`: 感知损失

4. **推理ODE步数**
   - 训练时不需要太多（模型内部用5-10步）
   - 推理时30-100步
   - 快速预览用30，最终结果用100

5. **恢复训练**
   ```yaml
   resume:
     checkpoint: "outputs_v2/checkpoints/checkpoint_epoch_50.pth"
   ```

---

## 🆚 V1 vs V2 选择

### **用V2（本模型）如果：**
- ✅ 有大量数据（>1000对）
- ✅ 推理时没有prompt
- ✅ 需要快速部署
- ✅ 端到端应用

### **用V1（原模型）如果：**
- ✅ 需要精确控制（指定目标位置）
- ✅ 数据较少（<500对）
- ✅ 研究和可解释性
- ✅ 有额外的prompt信息

---

## 📞 遇到问题？

### **检查清单**：
1. ✅ 数据目录结构正确？
2. ✅ sim和real文件名匹配？
3. ✅ 图像是灰度PNG？
4. ✅ `config_v2.yaml`路径正确？
5. ✅ GPU显存足够？（至少8GB）

### **Debug模式**：
```bash
# 测试数据加载
python utils_v2/dataset_v2.py

# 测试模型
python models_v2/flow_matching_v2.py

# 测试感知损失
python models_v2/perceptual_loss.py
```

---

## 🎓 进一步学习

- **README_V2.md**: 完整文档
- **config_v2.yaml**: 所有参数说明
- **models_v2/**: 模型实现细节

---

## ✨ 总结

V2模型的核心优势：
1. **简单**：不需要prompt，直接输入仿真图
2. **实用**：端到端，适合实际部署
3. **稳定**：Flow Matching + Perceptual Loss，训练稳定
4. **快速**：去掉ControlNet，推理更快

**开始你的第一次训练吧！** 🚀

```bash
python train_v2.py
```

