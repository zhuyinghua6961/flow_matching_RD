# Flow Matching RD图 Sim2Real

**基于Flow Matching的雷达RD图仿真到真实转换模型**

端到端的图像到图像转换，无需prompt或标注信息。

---

## 📊 模型架构

### 整体流程

```
输入: sim_image (仿真RD图, 512×512, 单通道)
  ↓
SimEncoder → 多尺度条件特征 [64, 128, 256, 512, 1024]
  ↓
初始化: x_0 ~ N(0, I) (随机噪声)
  ↓
Flow Matching ODE求解:
  for t = 0 → 1 (50步):
    v_t = ConditionalUNet(x_t, t, cond_features)
    x_{t+1} = x_t + v_t * dt
  ↓
输出: real_image (生成的真实RD图, 512×512)
```

### 核心组件

#### 1. SimEncoder（条件编码器）
- **作用**: 提取仿真图的结构和语义特征
- **架构**: 5层下采样 + ResNet Block
- **输出**: 多尺度特征金字塔，用于条件注入
- **参数量**: ~2M

#### 2. ConditionalUNet（条件去噪网络）
- **作用**: 在条件特征指导下，从噪声生成真实图
- **输入**: 噪声图 x_t + 时间步 t + 条件特征
- **架构**: UNet + Time Embedding + 跨层特征注入
- **输出**: 预测的速度场 v_t
- **参数量**: ~62M

#### 3. Flow Matching
- **训练**: 学习从噪声到真实图的最优传输路径
  ```python
  x_t = (1-t)·noise + t·real_image
  v_target = real_image - noise
  loss = MSE(model(x_t, t, sim), v_target)
  ```
- **推理**: ODE求解器沿学到的路径生成图像

**总参数量**: 64M

---

## 🎯 Loss函数设计

### 组合Loss

```python
Total Loss = Loss_FM + λ_freq·Loss_Freq + λ_ssim·Loss_SSIM
```

| Loss类型 | 权重 | 作用 | 原因 |
|---------|------|------|------|
| **Flow Matching Loss** | 1.0 | 基础MSE，学习整体映射 | 保证模型基本能力 |
| **频域Loss** | 2.5 | FFT频谱幅度匹配 | **核心！** 学习多普勒十字、杂波频率特征 |
| **SSIM Loss** | 0.5 | 结构相似性 | 辅助保持局部结构和纹理 |

### 为什么这样设计？

#### ❌ 不用感知损失（VGG Perceptual Loss）
- **原因**: VGG基于自然图像训练（ImageNet），无法理解雷达RD图的物理含义
- **问题**: 会学习自然图像的纹理和边缘，反而干扰雷达特征学习
- **实验**: 使用VGG后，模型倾向于生成"看起来平滑"的图像，但丢失多普勒十字

#### ✅ 频域Loss（核心）
- **原因**: 雷达RD图的核心是**多普勒效应**（频域特征）
  - 多普勒十字：运动目标在频域呈现十字形强响应
  - 杂波分布：高频成分反映杂波的空间分布
- **作用**: 直接在频域监督，强制模型学习物理层面的频率结构
- **实现**: 
  ```python
  # FFT变换到频域
  fft_pred = torch.fft.rfft2(pred, norm='ortho')
  fft_target = torch.fft.rfft2(target, norm='ortho')
  
  # 对数幅度谱（增强低频信息）
  mag_pred = torch.log(torch.abs(fft_pred) + 1e-8)
  mag_target = torch.log(torch.abs(fft_target) + 1e-8)
  
  # MSE Loss
  loss = F.mse_loss(mag_pred, mag_target)
  ```
- **为什么用对数**: 频谱幅度范围大（1-10000+），对数压缩后更易优化

#### ✅ SSIM Loss（辅助）
- **原因**: 保证局部结构和纹理的一致性
- **作用**: 
  - 像素级MSE关注全局，SSIM关注局部窗口
  - 捕捉人眼感知的结构相似性
- **权重**: 0.5（低于频域Loss），避免过度平滑

### 权重调整策略

```yaml
# 当前配置（频域主导）
frequency_weight: 2.5   # 主导训练，学习多普勒特征
ssim_weight: 0.5        # 辅助，保持结构
perceptual_weight: 0.0  # 关闭，对雷达图无效
```

**核心原则**: 
- 频域Loss权重 > SSIM权重，确保模型**优先学习物理特性**而非视觉纹理
- 如果多普勒十字不明显，可增大到3.0（极限值）
- 如果训练不稳定，降低到2.0

### Loss设计的物理意义

| 特征 | 空域（像素级） | 频域（FFT） |
|------|--------------|-----------|
| **多普勒十字** | 不明显，易被忽略 | ✅ 十字形强响应，非常显著 |
| **杂波分布** | 随机像素值 | ✅ 高频成分特征 |
| **目标位置** | ✅ 局部像素值 | 频域弥散 |

**结论**: 频域+空域结合，既学习物理特征（频域），又保证空间准确性（SSIM）。

---

## 📚 文档

详细使用说明请查看：
- [快速开始指南](docs/QUICKSTART.md) - 训练、推理、WebUI使用
- [数据准备](docs/DATA_PREPARATION.md) - 数据格式和预处理（如需要）
- [配置说明](config_v2.yaml) - 完整的配置参数注释

---

## 📈 核心优势

✅ **物理驱动**: 频域Loss直接约束多普勒效应，而非依赖视觉相似性  
✅ **端到端**: 无需prompt，直接从仿真到真实的映射学习  
✅ **快速收敛**: 频域特征梯度明确，训练更高效  
✅ **通用性强**: 适用于各类雷达RD图Sim2Real任务  

---

## 🎓 技术参考

- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **频域Loss**: 受傅里叶变换在信号处理中的应用启发
- **SSIM**: [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ieeexplore.ieee.org/document/1284395)
