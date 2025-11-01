# 🐛 推理结果全黑问题 - 根因分析与修复

## 问题描述

推理结果PSNR仅**9.36 dB**（正常应该>20 dB），图像几乎全黑：
- 推理结果：99.9%像素值 < 25（几乎全黑）
- 真实目标：主要像素值在75-100（正常）
- 均值差异：推理0.14 vs 真实86.21

## 根本原因

**WebUI插件的图像归一化/反归一化与训练时不一致！**

### 训练时的处理流程

```python
# utils_v2/dataset_v2.py (第174行)
transform = transforms.Compose([
    transforms.ToTensor(),                      # [0,255] -> [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5]) # [0,1] -> [-1,1]
])
```

### 修复前的推理流程（❌ 错误）

```python
# webui/backend/plugins/flow_matching_v2_plugin.py (第62-65行)
self.transform = transforms.Compose([
    transforms.Resize(self.image_size),
    transforms.ToTensor()  # ❌ 缺少 Normalize！
])

# 第301行
def _tensor_to_image(self, tensor):
    tensor = torch.clamp(tensor, 0, 1)  # ❌ 直接clamp，没有denormalize！
```

### 问题分析

| 阶段 | 训练时 | 修复前推理 | 问题 |
|------|--------|-----------|------|
| **输入** | [-1, 1] | [0, 1] | ❌ 分布不匹配 |
| **模型处理** | [-1, 1] → [-1, 1] | [0, 1] → [-1, 1] | ❌ 输入偏移 |
| **输出** | [-1, 1] | [-1, 1] | ✅ 一致 |
| **后处理** | (x+1)/2 → [0,1] | clamp(x,0,1) | ❌ 负值全变0 |
| **最终图像** | [0, 255] | [0, ~64] | ❌ 偏暗/全黑 |

**核心问题**：
1. 输入没有normalize到[-1,1]，导致模型输入分布错误
2. 输出直接clamp(x, 0, 1)，将所有负值（-1~0）裁剪成0，导致图像偏暗

**示例**：
```
模型输出 -0.5 （应该是灰色，像素值64）:
  ❌ 错误: clamp(-0.5, 0, 1) = 0.0  → 0   (黑色)
  ✅ 正确: (-0.5 + 1) / 2 = 0.25 → 64  (灰色)
```

## 修复方案

### 1. 添加输入归一化

```python
# webui/backend/plugins/flow_matching_v2_plugin.py (第61-66行)
self.transform = transforms.Compose([
    transforms.Resize(self.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # ✅ 添加这行
])
```

### 2. 添加输出反归一化

```python
# webui/backend/plugins/flow_matching_v2_plugin.py (第299-304行)
def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
    # tensor: (C, H, W), 范围 [-1, 1] (模型输出)
    # Denormalize: [-1, 1] -> [0, 1]
    tensor = (tensor + 1) / 2  # ✅ 添加这行
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    # ... 后续代码不变
```

## 修复后的正确流程

```
输入图像 [0, 255]
  ↓ ToTensor
[0, 1]
  ↓ Normalize(0.5, 0.5)
[-1, 1] ← 模型输入 ✅
  ↓ Model(x)
[-1, 1] ← 模型输出 ✅
  ↓ Denormalize: (x+1)/2
[0, 1]
  ↓ * 255
[0, 255] ← 输出图像 ✅
```

## 预期效果

| 指标 | 修复前 | 修复后（预期） |
|------|--------|---------------|
| PSNR | 9.36 dB | > 20 dB |
| 像素均值 | 0.14 | ~86 |
| 像素标准差 | 2.46 | ~12 |
| 视觉效果 | 几乎全黑 | 正常RD图 |

## 重新测试

修复后，请重新推理：

```bash
# 重启WebUI后端（使插件重新加载）
cd /home/user/桌面/flow_matching_RD/webui/backend
pkill -f "python main.py"
python main.py

# 或者在WebUI界面重新加载插件
# 点击 "Unload Model" -> "Load Model"

# 然后重新推理 rd1601.png
```

## 其他发现

从TensorBoard分析还发现：

1. **训练不充分**：波动率26%，仍在下降，建议继续训练20-30个epoch
2. **过拟合严重**：Train Loss (0.262) vs Val Loss (0.090)，差距192%（异常）
3. **数据集问题**：验证集Loss异常低，可能分布不均匀

这些问题需要继续优化训练，但**当前的全黑问题是由归一化bug导致的，与训练质量无关**。

## 总结

- ✅ 已修复：归一化/反归一化不一致
- ✅ 修复文件：`webui/backend/plugins/flow_matching_v2_plugin.py`
- ⚠️  需要重启WebUI后端或重新加载插件
- 📊 建议：继续训练模型20-30个epoch以获得更好效果

---

**Created**: 2025-11-01  
**Author**: AI Assistant  
**Status**: 已修复，待测试
