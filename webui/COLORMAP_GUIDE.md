# 🎨 伪彩色可视化指南

## 问题说明

### 为什么推理结果是灰度图？

**数据集真相**：
- 你的数据集中的"真实"RD图实际上是**应用了伪彩色映射（colormap）的可视化图像**
- RD图本质上是**单通道数据**（距离-速度），colormap只是为了更好的可视化
- 训练时使用`convert('L')`将RGB转为灰度，模型学到的是灰度信息
- 推理时模型输出灰度图，但可视化需要应用colormap

### 解决方案

在推理后应用相同的colormap，将灰度图转换为伪彩色图。

---

## 功能特性

### ✨ 自动应用Colormap

插件已自动添加colormap支持：
- ✅ **默认启用**：推理结果自动应用`jet` colormap
- ✅ **可配置**：支持15+种常见colormap
- ✅ **动态切换**：推理时可指定不同colormap
- ✅ **可关闭**：如需灰度图，可禁用colormap

---

## 使用方法

### 1️⃣ 默认使用（推荐）

插件默认配置已启用`jet` colormap：

```python
# webui/backend/main.py 注册插件时
config = {
    'checkpoint_path': 'outputs_v2/checkpoints/checkpoint_epoch_31.pth',
    'device': 'cuda:0',
    'base_channels': 64,
    'channel_mult': (1, 2, 4, 8),
    'apply_colormap': True,   # 默认启用
    'colormap_name': 'jet'     # 默认使用jet
}
```

推理结果会自动应用`jet` colormap，输出彩色图像。

---

### 2️⃣ 自定义Colormap

#### 通过配置指定

```python
config = {
    # ... 其他配置 ...
    'apply_colormap': True,
    'colormap_name': 'viridis'  # 使用viridis colormap
}
```

#### 推理时动态指定

```python
# API调用时传入colormap参数
result = plugin.inference(
    image_path='input.png',
    output_path='output.png',
    ode_steps=50,
    colormap='plasma'  # 临时使用plasma
)
```

---

### 3️⃣ 支持的Colormap列表

| Colormap | 风格 | 适用场景 |
|----------|------|----------|
| **jet** | 🌈 彩虹（蓝→绿→黄→红） | 默认，类似MATLAB |
| **viridis** | 🟣 紫→绿→黄 | 感知均匀，推荐科学可视化 |
| **plasma** | 🔵 蓝→紫→橙→黄 | 高对比度 |
| **inferno** | 🔴 黑→紫→橙→黄 | 暖色调 |
| **magma** | 🟠 黑→紫→红→白 | 暖色调 |
| **hot** | 🔥 黑→红→黄→白 | 热力图 |
| **cool** | ❄️ 青→紫 | 冷色调 |
| **turbo** | 🌈 改进版jet | Google Turbo |
| **gray** | ⚫ 灰度 | 不应用colormap |

完整列表：`jet`, `viridis`, `plasma`, `inferno`, `magma`, `hot`, `cool`, `spring`, `summer`, `autumn`, `winter`, `gray`, `bone`, `copper`, `turbo`

---

### 4️⃣ 禁用Colormap（输出灰度图）

如果需要原始灰度图：

```python
# 配置时禁用
config = {
    # ... 其他配置 ...
    'apply_colormap': False
}

# 或推理时指定
result = plugin.inference(
    image_path='input.png',
    output_path='output.png',
    apply_colormap=False
)
```

---

## 视觉效果对比

### Colormap效果示例

```
灰度图（原始）:
  ⬛⬛⬛⬛⬜⬜⬜⬜  → 单调，难以区分细节

Jet Colormap:
  🔵🔵🟢🟡🟠🔴🔴🔴  → 彩虹，高对比度，经典

Viridis Colormap:
  🟣🟣🔵🟢🟡🟡🟡🟡  → 感知均匀，护眼

Hot Colormap:
  ⬛⬛🟣🔴🟠🟡⬜⬜  → 热力图，温度感
```

### RD图实际应用

- **低速目标区域**：蓝色/紫色（低值）
- **中速目标区域**：绿色/黄色（中值）
- **高速目标区域**：橙色/红色（高值）

通过颜色梯度，可以更直观地识别目标速度分布。

---

## API参数说明

### `inference()` 方法新增参数

```python
def inference(
    image_path: str,
    output_path: str,
    ode_steps: int = 50,
    ode_method: str = 'euler',
    apply_colormap: bool = None,  # ✨ 新增
    colormap: str = None,         # ✨ 新增
    **kwargs
) -> Dict[str, Any]
```

**参数说明**：
- `apply_colormap`: 是否应用伪彩色（`None`=使用配置，`True`=强制启用，`False`=强制禁用）
- `colormap`: colormap名称（`None`=使用配置，其他=临时使用指定colormap）

---

## 高级用法

### 1. 对比不同Colormap

```python
colormaps = ['jet', 'viridis', 'plasma', 'hot']

for cmap in colormaps:
    plugin.inference(
        image_path='input.png',
        output_path=f'output_{cmap}.png',
        colormap=cmap
    )
```

### 2. 同时保存灰度和彩色

```python
# 保存灰度图
plugin.inference(
    image_path='input.png',
    output_path='output_gray.png',
    apply_colormap=False
)

# 保存彩色图
plugin.inference(
    image_path='input.png',
    output_path='output_color.png',
    apply_colormap=True,
    colormap='jet'
)
```

### 3. 批量推理保持colormap

```python
# 批量推理会自动使用配置的colormap
plugin.batch_inference(
    image_paths=['img1.png', 'img2.png', 'img3.png'],
    output_dir='outputs/',
    ode_steps=50
)
```

---

## 技术原理

### 数据流程

```
训练阶段:
  RGB伪彩色图 (数据集)
    ↓ convert('L')
  灰度图 (训练数据)
    ↓ 模型学习
  灰度→灰度 映射

推理阶段:
  灰度输入
    ↓ 模型推理
  灰度输出
    ↓ apply_colormap (修复)
  RGB伪彩色图 (可视化)
```

### Colormap实现

```python
# 使用matplotlib的colormap
import matplotlib.cm as cm

colormap = cm.get_cmap('jet')
normalized = gray_image / 255.0  # [0, 255] → [0, 1]
colored = colormap(normalized)   # 应用colormap
rgb = (colored[:,:,:3] * 255).astype(np.uint8)  # RGB输出
```

---

## FAQ

### Q1: 为什么数据集是RGB但训练用灰度？

**A**: RD图本质是单通道数据（强度值），RGB只是可视化手段。训练时转灰度可以：
- 减少模型复杂度（1通道 vs 3通道）
- 学习真实的强度关系，而非colormap风格
- 推理时可灵活选择不同colormap

### Q2: 不同colormap会影响推理质量吗？

**A**: 不会。colormap只是后处理的可视化，不影响模型输出的灰度值。就像给黑白照片上色，本质信息不变。

### Q3: 如何让输出完全匹配数据集风格？

**A**: 使用`jet` colormap（默认），这是最接近MATLAB/常见雷达可视化的风格。如果数据集使用其他colormap，请相应调整。

### Q4: Colormap是在GPU还是CPU上计算？

**A**: CPU。Colormap应用是后处理步骤，在CPU上用numpy完成，不占用GPU资源。

### Q5: 批量推理能用不同colormap吗？

**A**: 可以。在for循环中对每张图单独调用`inference()`并指定不同的`colormap`参数。

---

## 总结

✅ **问题已解决**：
- 推理结果不再是单调的灰度图
- 自动应用伪彩色，匹配数据集风格
- 支持15+种colormap，灵活可配置

🎨 **推荐配置**：
```python
config = {
    'apply_colormap': True,
    'colormap_name': 'jet'  # 或 'viridis' (更现代)
}
```

📚 **相关文档**：
- `BUGFIX_NORMALIZATION.md` - 归一化问题修复
- `PLUGIN_GUIDE.md` - 插件开发指南

---

**Created**: 2025-11-01  
**Author**: AI Assistant  
**Status**: 已实现并测试
