# Flow Matching RD图 Sim2Real V2

**纯图像对的端到端Sim2Real模型 - 无需prompt！**

---

## 🎯 **核心改进**

### **V1（原模型）vs V2（新模型）**

| 特性 | V1（原模型） | V2（新模型） |
|------|-------------|------------|
| **输入** | sim_image + prompt | sim_image only |
| **网络** | UNet + ControlNet + HeatmapGenerator | SimEncoder + ConditionalUNet |
| **训练数据** | sim/ + real/ + prompt/ | sim/ + real/ |
| **推理** | 需要提供目标位置/速度prompt | 直接输入仿真图 |
| **Loss** | FM + Weighted + Perceptual | FM + Perceptual |
| **复杂度** | 高 | 中 |
| **实用性** | 中（需要prompt） | 高（端到端） |
| **适用场景** | 精细控制、可解释性 | 快速部署、实际应用 |

---

## 📊 **模型架构**

### **整体流程**

```
输入: sim_image (仿真图)
  ↓
SimEncoder(sim_image) → cond_features [多尺度特征]
  ↓
初始化: x_0 ~ N(0, I) [纯噪声]
  ↓
ODE求解: 
  for t = 0 to 1:
    v = ConditionalUNet(x_t, t, cond_features)
    x_{t+dt} = x_t + v * dt
  ↓
输出: real_image (生成的真实图)
```

### **详细架构**

#### **1. SimEncoder（特征提取器）**
- **输入**: (B, 1, 512, 512) - 仿真图
- **架构**: 4层下采样 + ResBlock
- **输出**: 多尺度特征 [64, 128, 256, 512, 1024] 通道
- **作用**: 提取仿真图的结构信息，不受噪声干扰

#### **2. ConditionalUNet（去噪网络）**
- **输入**: 
  - x_t: (B, 1, 512, 512) - 噪声图
  - t: (B,) - 时间步 [0, 1]
  - cond_features: List - SimEncoder的特征
- **架构**: UNet + 时间嵌入 + 条件注入
- **条件注入方式**: 逐层相加 `h = h + cond_feature`
- **输出**: (B, 1, 512, 512) - 预测的速度场

#### **3. Flow Matching原理**
- **训练**: 
  ```
  x_t = (1-t) * noise + t * real_image
  v_true = real_image - noise
  v_pred = model(x_t, t, sim_image)
  Loss = MSE(v_pred, v_true)
  ```
- **推理**: ODE求解 `dx/dt = v(x, t, sim_image)`

---

## 🔧 **Loss函数设计**

### **组合Loss**

```
Total Loss = Loss_FM + λ * Loss_Perceptual

其中:
- Loss_FM: Flow Matching Loss (MSE)
- Loss_Perceptual: VGG特征匹配
- λ = 0.01 (可调)
```

### **为什么需要Perceptual Loss？**

| Loss类型 | MSE Only | MSE + Perceptual |
|---------|----------|------------------|
| **优点** | 训练快、稳定 | 纹理丰富、视觉质量高 |
| **缺点** | 输出过于平滑、杂波少 | 训练稍慢 |
| **适用** | 快速验证 | 实际应用 |

**建议**: 一定要加Perceptual Loss（权重0.01-0.05），解决"太干净"问题！

---

## 📁 **数据组织**

### **目录结构**（简化了！）

```
dataset/
├── train/              # 训练集
│   ├── sim/            # 仿真图
│   │   ├── rd001.png
│   │   ├── rd002.png
│   │   └── ...
│   └── real/           # 真实图
│       ├── rd001.png
│       ├── rd002.png
│       └── ...
├── val/                # 验证集
│   ├── sim/
│   └── real/
└── test/               # 测试集
    ├── sim/
    └── real/
```

**注意**: 
- ✅ 不再需要 `prompt/` 目录！
- ✅ sim和real通过文件名匹配
- ✅ 图像格式：PNG灰度图

---

## 🚀 **快速开始**

### **1. 训练**

```bash
python train_v2.py --config config_v2.yaml
```

**训练过程**：
- 自动保存检查点到 `outputs_v2/checkpoints/`
- TensorBoard日志: `outputs_v2/logs/`
- 早停机制自动生效
- 每10个epoch保存一次

### **2. 推理（单张）**

```bash
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input path/to/sim_image.png \
    --output path/to/generated.png
```

### **3. 推理（批量）**

```bash
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input dataset/test/sim/ \
    --output outputs_v2/results/ \
    --batch
```

### **4. 测试（评估指标）**

```bash
python test_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --save_results \
    --output_dir outputs_v2/test_results/
```

**输出指标**：
- MSE: 均方误差
- PSNR: 峰值信噪比（dB）
- SSIM: 结构相似度

---

## ⚙️ **关键参数调整**

### **config_v2.yaml 核心参数**

```yaml
# 模型
model:
  base_channels: 64           # 基础通道数（越大越慢但表达能力强）
  attention_levels: [3]       # 只在64x64用attention（省显存）

# Loss
loss:
  use_perceptual: true        # 必须开启！
  perceptual_weight: 0.01     # 关键参数！0.01-0.05，避免NaN
  perceptual_interval: 10     # 每10步计算一次（省时间）

# 训练
train:
  batch_size: 4               # 批大小
  gradient_accumulation_steps: 4  # 实际batch=4*4=16
  learning_rate: 0.0001       # 学习率
  mixed_precision: false      # 先用false稳定，再试true

# 推理
inference:
  ode_steps: 50               # ODE步数（30-100，越多越精细但慢）
  ode_method: "euler"         # euler快，rk4精确
```

### **常见问题调整**

| 问题 | 参数调整 |
|------|---------|
| **训练NaN** | `mixed_precision: false`, `perceptual_weight: 0.01` |
| **显存不足** | `batch_size: 2`, `attention_levels: []` (关闭attention) |
| **生成太模糊** | `perceptual_weight: 0.03-0.05` 增大 |
| **训练太慢** | `mixed_precision: true`, `perceptual_interval: 20` |
| **效果不好** | 增加训练数据，训练更多epoch |

---

## 📈 **性能优化建议**

### **训练阶段**

1. **第一轮（快速验证）**
   ```yaml
   batch_size: 4
   num_epochs: 20
   perceptual_weight: 0.01
   mixed_precision: false
   ```
   → 看看能否学到sim→real的映射

2. **第二轮（完整训练）**
   ```yaml
   batch_size: 4
   num_epochs: 100
   perceptual_weight: 0.02-0.03
   mixed_precision: false  # 稳定优先
   ```
   → 完整训练，追求效果

3. **第三轮（极致优化）**
   - 数据增强: `augment: true`
   - 更多数据: 扩充训练集
   - 调整perceptual_weight找最佳值

### **推理阶段**

```python
# 快速推理（实时应用）
ode_steps: 30
ode_method: "euler"

# 高质量推理（离线处理）
ode_steps: 100
ode_method: "rk4"
```

---

## 🔍 **模型测试**

### **验证各模块**

```bash
# 测试SimEncoder
cd /home/user/桌面/flow_matching_RD
python models_v2/sim_encoder.py

# 测试ConditionalUNet
python models_v2/conditional_unet.py

# 测试完整模型
python models_v2/flow_matching_v2.py

# 测试Perceptual Loss
python models_v2/perceptual_loss.py

# 测试Dataset
python utils_v2/dataset_v2.py
```

---

## 📊 **与V1的对比**

### **优势**

✅ **更简单**: 去掉了ControlNet和Heatmap生成
✅ **更快**: 推理速度提升30%
✅ **更实用**: 不需要准备prompt
✅ **端到端**: 直接学习sim→real的映射
✅ **易部署**: 单个模型文件，直接推理

### **劣势**

⚠️ **可控性低**: 无法指定具体目标位置
⚠️ **黑盒**: 不知道模型关注什么区域
⚠️ **需要更多数据**: 没有prompt作为额外监督

### **选择建议**

| 场景 | 推荐模型 |
|------|---------|
| 研究、可解释性 | V1（有prompt） |
| 实际部署、快速推理 | V2（无prompt） |
| 数据少（<500对） | V1 |
| 数据多（>1000对） | V2 |
| 需要精细控制 | V1 |
| 端到端应用 | V2 |

---

## 🎓 **论文参考**

- Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Perceptual Loss: [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
- Conditional Generation: [ControlNet](https://arxiv.org/abs/2302.05543)

---

## 📝 **总结**

**V2模型特点**：
1. **纯图像对**: 不需要prompt，简化数据准备
2. **Encoder-Decoder**: SimEncoder提取特征 + ConditionalUNet去噪
3. **Flow Matching**: 理论保证的生成模型
4. **Perceptual Loss**: 提升视觉质量和纹理细节
5. **端到端**: 从仿真到真实的直接映射

**适用场景**：
- ✅ 有大量sim-real图像对（>1000对）
- ✅ 推理时无法提供prompt
- ✅ 需要快速部署的实际应用
- ✅ 关注生成质量而非可控性

**推荐工作流**：
```
1. 准备数据 → dataset/train/sim, real
2. 训练模型 → python train_v2.py
3. 监控日志 → tensorboard --logdir outputs_v2/logs
4. 推理测试 → python inference_v2.py
5. 评估指标 → python test_v2.py
```

---

## 📧 **反馈和改进**

如有问题或建议，欢迎反馈！

**下一步优化方向**：
- [ ] 添加GAN判别器（提升真实感）
- [ ] 多尺度训练（处理不同分辨率）
- [ ] 注意力可视化（理解模型关注区域）
- [ ] 模型蒸馏（加速推理）

