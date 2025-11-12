# V4 快速开始指南

## 🚀 快速上手（5分钟）

### 步骤1：预训练基础模型（如果还没有）

```bash
# 使用V2/V3训练基础模型
python train_v3.py --config config_v2.yaml

# 等待训练完成，得到：
# trained_models/outputs/checkpoints/best_model.pth
```

### 步骤2：配置预训练模型路径

编辑 `v4_finetune/config_finetune.yaml`，设置预训练模型路径：
```yaml
paths:
  pretrained_model: "../trained_models/outputs/checkpoints/best_model.pth"
```

### 步骤3：微调多普勒效应

```bash
# 方式1（推荐）：使用配置文件中的路径
python v4_finetune/train_finetune.py --config v4_finetune/config_finetune.yaml

# 方式2：命令行指定（会覆盖配置文件）
python v4_finetune/train_finetune.py \
    --config v4_finetune/config_finetune.yaml \
    --pretrained /path/to/your/model.pth

# 快速收敛（约50个epoch）
# 输出：trained_models/v4_finetuned/checkpoints/best_finetuned.pth
```

### 步骤4：测试效果

```bash
# 方式1（推荐）：自动使用final_finetuned.pth
python v4_finetune/test_finetuned.py \
    --config v4_finetune/config_finetune.yaml \
    --save_results

# 方式2：指定特定模型
python v4_finetune/test_finetuned.py \
    --checkpoint trained_models/v4_finetuned/checkpoints/best_finetuned.pth \
    --save_results
```

---

## 📋 完整命令示例

### 场景1：从头开始（完整流程）

```bash
# 1. 预训练基础模型
python train_v3.py --config config_v2.yaml

# 2. 编辑配置文件，设置预训练模型路径
# 编辑 v4_finetune/config_finetune.yaml:
#   paths:
#     pretrained_model: "../trained_models/outputs/checkpoints/best_model.pth"

# 3. 微调多普勒（直接使用配置文件）
python v4_finetune/train_finetune.py --config v4_finetune/config_finetune.yaml

# 4. 测试评估（自动使用final_finetuned.pth）
python v4_finetune/test_finetuned.py \
    --config v4_finetune/config_finetune.yaml \
    --save_results
```

### 场景2：已有预训练模型（只微调）

```bash
# 方式1：修改配置文件中的预训练模型路径，然后直接运行
python v4_finetune/train_finetune.py --config v4_finetune/config_finetune.yaml

# 方式2：命令行指定（不修改配置文件）
python v4_finetune/train_finetune.py \
    --config v4_finetune/config_finetune.yaml \
    --pretrained /path/to/your/pretrained_model.pth
```

### 场景3：自定义配置

```bash
# 1. 复制并修改配置文件
cp v4_finetune/config_finetune.yaml v4_finetune/my_config.yaml
# 编辑 my_config.yaml，调整参数

# 2. 使用自定义配置微调
python v4_finetune/train_finetune.py \
    --config v4_finetune/my_config.yaml \
    --pretrained trained_models/outputs/checkpoints/best_model.pth
```

---

## ⚙️ 关键配置说明

### 微调学习率（最重要）

```yaml
finetune:
  lr_generator: 1e-5      # 生成器学习率（推荐1e-5到2e-5）
  lr_discriminator: 2e-4  # 判别器学习率（推荐1e-4到3e-4）
```

**调参建议**：
- 如果多普勒改善慢：提高`lr_generator`到`2e-5`
- 如果背景被影响：降低`lr_generator`到`5e-6`

### GAN权重

```yaml
finetune:
  gan_weight: 0.5              # GAN总权重（推荐0.3-0.8）
  adversarial_weight: 1.0      # 对抗损失权重
  feature_matching_weight: 1.0 # 特征匹配权重
```

**调参建议**：
- 如果多普勒效果不明显：提高`gan_weight`到`0.8`
- 如果训练不稳定：降低`gan_weight`到`0.3`

### 参数冻结模式

```yaml
finetune:
  freeze_mode: "selective"  # 选项：selective, freeze_encoder, all_trainable
```

**模式说明**：
- `selective`：选择性微调（**推荐**⭐）- 冻结编码器+低频层
- `freeze_encoder`：只冻结编码器 - 适度微调
- `all_trainable`：所有参数可训练 - 不推荐

---

## 📊 监控训练

### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir trained_models/v4_finetuned/logs

# 打开浏览器访问：http://localhost:6006
```

### 重点关注指标

**训练日志中**：
```
Epoch 10:
  生成器损失: 0.1234      # 应该稳定下降
  判别器损失: 0.5678      # 应该稳定在0.3-0.7
  判别器准确率: Real=0.68, Fake=0.42  # 理想状态50-60%
```

**TensorBoard中**：
- `train/loss_adversarial` - 对抗损失（应该下降）
- `train/d_real_acc` - 判别真实图的准确率（60-70%）
- `train/d_fake_acc` - 判别生成图的准确率（30-40%）

---

## 🔧 常见问题快速解决

### 问题1：判别器太强（D_acc > 70%）

**现象**：生成器学不到东西

**快速解决**：
```bash
# 方案1：降低判别器学习率
# 修改config中：lr_discriminator: 1e-4

# 方案2：降低判别器更新频率
# 修改config中：discriminator_update_freq: 2
```

### 问题2：判别器太弱（D_acc < 30%）

**现象**：无法区分真假

**快速解决**：
```bash
# 方案1：提高判别器学习率
# 修改config中：lr_discriminator: 3e-4

# 方案2：降低dropout
# 修改config中：dropout: 0.2
```

### 问题3：背景被影响了

**现象**：生成图的背景和预训练模型不一样

**快速解决**：
```bash
# 方案1：降低生成器学习率
# 修改config中：lr_generator: 5e-6

# 方案2：改用freeze_encoder模式
# 修改config中：freeze_mode: "freeze_encoder"
```

### 问题4：多普勒改善不明显

**现象**：微调后多普勒效果提升有限

**快速解决**：
```bash
# 方案1：提高GAN权重
# 修改config中：gan_weight: 0.8

# 方案2：提高生成器学习率
# 修改config中：lr_generator: 2e-5

# 方案3：延长训练
# 修改config中：num_epochs: 100
```

---

## 📈 预期训练时间

**硬件配置**：NVIDIA RTX 3090 / 4090

**数据集规模**：约200-300张图片

**预期时间**：
- 预训练阶段（V2/V3）：8-12小时（200个epoch）
- 微调阶段（V4）：**1-2小时**（50个epoch）✅
- 总计：10-14小时

**如果使用较弱GPU**：
- RTX 3060/3070：时间×1.5-2
- RTX 2080Ti：时间×2-3

---

## ✅ 成功标志

### 微调成功的标志

1. **判别器准确率稳定在50-60%**
   - Real_acc ≈ 65-70%
   - Fake_acc ≈ 35-45%

2. **频域相关系数提升**
   - 微调前：0.70-0.75
   - 微调后：0.82-0.88 ✅

3. **背景保持不变**
   - 对比微调前后的生成图
   - 背景应该几乎一致

4. **多普勒十字更清晰**
   - 视觉检查生成图
   - 多普勒十字应该更明显

---

## 🎯 下一步

### 微调完成后

1. **对比分析**
   ```bash
   # 测试预训练模型
   python test_v3.py --checkpoint pretrained_model.pth
   
   # 测试微调模型
   python v4_finetune/test_finetuned.py --checkpoint finetuned_model.pth
   
   # 对比两者的频域相关系数
   ```

2. **可视化对比**
   - 查看`v4_test_results/`中的对比图
   - 重点关注多普勒十字区域
   - 确认背景没有变化

3. **进一步优化**
   - 根据测试结果调整配置
   - 可以尝试不同的`freeze_mode`
   - 可以调整GAN权重

---

## 💡 专家建议

### 第一次使用

1. **使用默认配置**
   - 不要修改配置，直接运行
   - 观察训练过程和结果
   - 建立基准

2. **监控判别器准确率**
   - 应该快速达到50-60%并稳定
   - 如果偏离太多，及时调整

3. **定期检查生成结果**
   - 每10个epoch保存一次检查点
   - 生成几张图看效果
   - 确认方向正确

### 调优技巧

1. **小步快跑**
   - 每次只调整一个参数
   - 观察效果后再调整其他
   - 记录每次调整的结果

2. **保存实验记录**
   - 记录每次实验的配置
   - 记录最终的评分
   - 方便回溯和对比

3. **关注核心指标**
   - 频域相关系数（最重要）
   - 判别器准确率（稳定性指标）
   - 背景一致性（视觉检查）

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志**：检查训练日志中的错误信息
2. **检查配置**：确认配置文件格式正确
3. **查看README**：v4_finetune/README.md有详细说明
4. **调试模式**：设置`num_epochs: 5`快速测试

---

**祝训练顺利！** 🎉
