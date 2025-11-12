# Train V3 训练全流程

## 一、初始化阶段

1. **加载配置**：读取YAML配置文件
2. **环境设置**：随机种子、CuDNN、设备选择
3. **创建目录**：输出目录、日志目录、检查点目录
4. **创建模型**：Sim2RealFlowModel，可加载预训练权重
5. **数据加载**：创建训练集、验证集、测试集（可选）DataLoader
6. **优化器设置**：优化器、学习率调度器、混合精度
7. **训练设置**：早停机制、梯度累积
8. **GAN设置**（可选）：判别器初始化
9. **初始化PSNR历史**：用于动态范围计算

## 二、训练循环（每个Epoch）

### 1. 训练阶段
- 模型设置为训练模式
- 遍历训练批次：
  - 计算Flow Matching Loss（核心损失）
  - 生成预测样本（用于结构loss计算）
  - 计算Perceptual Loss（可选）
  - 计算频域Loss（学习多普勒结构）
  - 计算SSIM Loss（结构相似性）
  - 计算GAN Loss（可选，多普勒+地杂波差别）
  - 计算总损失
  - 反向传播
  - 梯度累积和参数更新
  - 记录日志到TensorBoard

### 2. 验证阶段
- 模型设置为评估模式
- 遍历验证批次：
  - 计算各项损失（无梯度）
  - **关键**：计算PSNR并添加到历史记录
  - 计算频域Loss、SSIM Loss、GAN Loss
- 记录PSNR历史到TensorBoard
- 记录验证指标到TensorBoard

### 3. 后处理
- 记录epoch级别指标到TensorBoard
- 学习率调度（根据调度器类型）
- **检查最佳模型**：
  - 如果验证loss < 最佳loss：立即保存best_model.pth
- **定期保存检查点**：
  - 每隔save_interval个epoch保存一次（如果该epoch不是最佳模型）
- **早停检查**：
  - 如果连续patience个epoch没有改善（改善 < min_delta）
  - 触发早停：加载最佳模型，保存final_model.pth，退出训练

## 三、训练结束处理

### 1. 正常结束（未触发早停）
- 加载最佳模型
- 保存final_model.pth（使用最佳模型权重）

### 2. 计算PSNR范围
- 基于验证历史数据计算P5-P95百分位数
- 计算min、max、mean、std
- 保存到检查点

### 3. 自动测试评估（如果启用）
- 遍历测试集
- 使用ODE求解生成图像
- 计算各项指标：
  - 频域相关系数
  - 频域MSE、PSNR
  - SSIM
  - PSNR
  - MAE
  - 能量分布相似度
- 计算综合评分：
  - 总分 = 60% × 频域得分 + 25% × SSIM得分 + 15% × PSNR得分
- 自动评级：优秀/良好/一般/需改进/较差
- 输出评分报告

## 四、关键机制

### 最佳模型保存
- **立即保存**：检测到最佳模型时立即保存，不依赖save_interval
- **确保不丢失**：最佳模型始终会被保存到best_model.pth

### 最终模型保存
- **早停时**：使用最佳模型的权重保存final_model.pth
- **正常结束时**：使用最佳模型的权重保存final_model.pth
- **关键**：final_model.pth保存的是最佳模型，而不是最后一个epoch的模型

### PSNR动态范围
- **训练时收集**：每个epoch验证时收集PSNR值
- **动态计算**：基于历史数据计算P5-P95范围
- **保存到检查点**：用于测试时的归一化评分

### 早停机制
- **判断逻辑**：如果连续patience个epoch，验证loss没有改善（改善 < min_delta），触发早停
- **使用最佳模型**：早停时确保使用最佳模型权重，而不是当前epoch的模型

### GAN支持（可选）
- **多普勒+地杂波判别器**：提取特征并计算差别
- **可选训练**：判别器可以只用于特征提取，不参与训练
- **权重配置**：多普勒0.75，地杂波0.25

## 五、输出文件

### 检查点目录
- `best_model.pth`：最佳模型（验证loss最低）
- `final_model.pth`：最终模型（使用最佳模型权重）
- `checkpoint_epoch_*.pth`：定期检查点（保留最近N个）

### 日志目录
- TensorBoard日志文件

## 六、监控指标

### TensorBoard指标
- `train/loss`：训练总损失
- `train/loss_fm`：Flow Matching损失
- `train/loss_frequency`：频域损失
- `train/loss_ssim`：SSIM损失
- `train/loss_gan`：GAN损失（如果启用）
- `val/loss`：验证总损失
- `val/psnr`：验证PSNR
- `epoch/train_loss`：epoch级别训练损失
- `epoch/val_loss`：epoch级别验证损失
- `test/*`：测试集指标（如果启用自动测试）

## 七、使用流程

1. **准备配置文件**：设置训练参数、数据路径等
2. **启动训练**：运行train_v3.py
3. **监控训练**：使用TensorBoard查看训练曲线
4. **等待完成**：训练自动完成或触发早停
5. **查看结果**：检查best_model.pth和final_model.pth
6. **测试评估**：如果启用自动测试，会输出综合评分

## 八、关键特性总结

- ✅ 最佳模型立即保存机制
- ✅ 最终模型使用最佳权重
- ✅ PSNR动态范围计算
- ✅ 自动测试评估
- ✅ 早停机制优化
- ✅ GAN支持（可选）
- ✅ 完整的日志记录