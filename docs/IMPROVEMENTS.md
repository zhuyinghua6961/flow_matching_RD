# 模型改进建议

## 🎯 **核心问题**

### 问题1：推理结果太干净，缺少真实杂波
- **原因**：`weight_factor=50`过高，模型只关注目标区域，忽略背景
- **解决**：降低到`weight_factor=10`，让模型学习背景细节

### 问题2：推理时不用prompt会影响效果
- **原因**：训练时依赖精确heatmap，推理时自动检测不一致
- **解决**：训练时20%概率使用均匀heatmap，减少依赖

---

## ✅ **已实施的改进**

### 1. **训练策略改进** (`train.py`)
```python
# 20%的概率不使用精确heatmap
if np.random.random() < 0.2:
    heatmap = torch.ones(...) * 0.3  # 均匀热力图
else:
    heatmap = self.heatmap_generator(prompt)  # 精确热力图
```

**效果**：
- ✅ 模型学会从sim图本身推断目标
- ✅ 推理时不依赖精确prompt也能工作
- ✅ 提升泛化性

### 2. **Loss权重调整** (`config.yaml`)
```yaml
loss:
  weight_factor: 10        # 从50降到10
  use_perceptual: true     # 启用感知损失
  perceptual_weight: 0.1
```

**效果**：
- ✅ 背景区域loss权重提高5倍
- ✅ 模型开始学习杂波和纹理细节
- ✅ 感知损失提升视觉真实感

### 3. **验证集集成** (`train.py`)
```python
# 每个epoch后在验证集评估
val_losses = self.validate_one_epoch(epoch)

# 使用验证loss做早停
if val_losses is not None:
    current_loss = val_losses['loss']
```

**效果**：
- ✅ 避免过拟合
- ✅ 更好的泛化性能
- ✅ 及时发现训练问题

---

## 📊 **预期效果对比**

| 指标 | 修改前 | 修改后（预期） |
|------|--------|----------------|
| 目标保真度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 背景杂波学习 | ⭐ | ⭐⭐⭐⭐ |
| 推理无prompt可用性 | ❌ | ✅ |
| 真实感 | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 🚀 **重新训练步骤**

### 1. **清理旧模型**
```bash
# 备份旧检查点
mv checkpoints checkpoints_old

# 重新开始训练
python train.py
```

### 2. **监控训练**
```bash
# 另一个终端
tensorboard --logdir=./logs
```

**关注指标**：
- `epoch/train_loss` vs `epoch/val_loss` - 观察是否过拟合
- `epoch/train_bg_loss` - 背景loss应该明显下降
- `epoch/val_loss` - 验证性能

### 3. **训练建议**
- **训练轮数**：至少50 epoch，观察验证loss曲线
- **早停**：如果20 epoch验证loss无改善，自动停止
- **最佳模型**：`checkpoints/best_model.pth`

---

## 🧪 **测试新模型**

### **无prompt推理**
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --sim_rd dataset/test/sim/rd1601.png \
  --output results/new_result.png \
  --visualize
```

**观察点**：
1. 背景杂波是否增加？
2. 目标位置是否准确？
3. 整体真实感如何？

### **有prompt推理（对比）**
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --sim_rd dataset/test/sim/rd1601.png \
  --prompt "radar-RD-map; ... target number = 1, ..." \
  --output results/with_prompt.png
```

### **定量评估**
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --save_results \
  --output_dir test_results_new
```

**关注指标**：
- **PSNR**: 应该略微下降（因为添加了杂波）
- **SSIM**: 应该提升（更真实的纹理）
- **视觉质量**: 主观评价更重要

---

## 🔧 **进一步调优（如果效果仍不理想）**

### 选项1：进一步降低weight_factor
```yaml
loss:
  weight_factor: 5  # 更关注背景
```

### 选项2：增加均匀heatmap概率
```python
if np.random.random() < 0.4:  # 从0.2改到0.4
    heatmap = torch.ones(...) * 0.3
```

### 选项3：调整感知损失权重
```yaml
loss:
  use_perceptual: true
  perceptual_weight: 0.2  # 从0.1提高到0.2
```

### 选项4：增加训练数据多样性
如果数据集中sim图本身就缺少变化，可以考虑：
- 收集更多样的仿真数据
- 或者在sim图上添加轻微噪声增强

---

## 📝 **训练日志解读**

### 正常的训练曲线应该是：
```
Epoch 1:   [Train] Loss: 0.0523, Target: 0.0421, BG: 0.0102
           [Val]   Loss: 0.0498 ⭐ (最佳)
           
Epoch 10:  [Train] Loss: 0.0234, Target: 0.0176, BG: 0.0058
           [Val]   Loss: 0.0251 ⭐ (最佳)
           
Epoch 30:  [Train] Loss: 0.0145, Target: 0.0098, BG: 0.0047
           [Val]   Loss: 0.0162 ⭐ (最佳)
```

**异常情况**：
- ⚠️ `BG Loss`不下降 → 背景没有学习，检查`weight_factor`
- ⚠️ `Train Loss << Val Loss` → 过拟合，需要更多正则化
- ⚠️ `Val Loss`震荡 → 学习率可能太大

---

## ❓ **常见问题**

### Q1: 训练多久能看到效果？
A: 通常10-20个epoch就能看到明显改善，最佳模型可能在30-50 epoch。

### Q2: 如果背景还是太干净怎么办？
A: 
1. 继续降低`weight_factor`到5
2. 检查真实数据是否确实有杂波
3. 考虑使用更复杂的loss（如GAN loss）

### Q3: 目标位置变得不准确怎么办？
A: `weight_factor`降太多了，回调到15-20。

### Q4: 推理无prompt效果很差？
A: 提高训练时均匀heatmap概率到30-40%。

---

## 📌 **总结**

**关键修改**：
1. ✅ `weight_factor: 50 → 10` 
2. ✅ `use_perceptual: false → true`
3. ✅ 训练时20%使用均匀heatmap
4. ✅ 集成验证集和早停

**预期结果**：
- 推理结果会有更多真实杂波
- 推理时可以不依赖prompt
- 整体视觉效果更真实

**下一步**：
重新训练模型，监控验证loss，观察背景杂波学习情况！

