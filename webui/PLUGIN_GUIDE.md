# 📝 插件开发指南

本指南将教你如何为 Sim2Real WebUI 开发自定义推理插件。

---

## 🎯 **核心概念**

插件是实现了 `InferenceInterface` 接口的Python类，WebUI通过这个统一接口与各种Sim2Real模型交互。

### 为什么需要插件？

- ✅ **解耦**：模型逻辑与WebUI分离
- ✅ **扩展**：新增模型无需修改WebUI代码
- ✅ **复用**：同一个WebUI支持所有Sim2Real任务

---

## 🚀 **快速开始**

### 1. 复制模板

```bash
cd webui/backend/plugins
cp plugin_template.py my_model_plugin.py
```

### 2. 修改类名和配置

```python
class MyModelPlugin(InferenceInterface):
    """
    你的模型插件
    
    必需配置参数:
        checkpoint_path: str - 模型检查点路径
        device: str - 设备
    """
    
    def __init__(self, plugin_name: str, config: Dict[str, Any]):
        super().__init__(plugin_name, config)
        
        # 添加你的配置参数
        self.your_param = config.get('your_param', default_value)
```

### 3. 实现必需方法

你必须实现以下5个抽象方法：

#### (1) `load_model()` - 加载模型

```python
def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
    """加载模型到GPU/CPU"""
    try:
        # 1. 创建模型
        self.model = YourModel()
        
        # 2. 加载权重
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 3. 移动到设备
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # 4. 标记已加载
        self.is_loaded = True
        
        print(f"✅ 模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        self.is_loaded = False
        return False
```

#### (2) `unload_model()` - 卸载模型

```python
def unload_model(self) -> bool:
    """释放显存"""
    try:
        if self.model is not None:
            del self.model
            self.model = None
        
        torch.cuda.empty_cache()
        self.is_loaded = False
        
        print("✅ 模型卸载成功")
        return True
    except Exception as e:
        print(f"❌ 卸载失败: {e}")
        return False
```

#### (3) `inference()` - 单张推理

```python
@torch.no_grad()
def inference(
    self, 
    image_path: str,
    output_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    单张图片推理
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        **kwargs: 前端传入的自定义参数
    
    Returns:
        {
            'success': bool,
            'output_path': str,
            'inference_time': float,
            'message': str,
            'metadata': dict  # 可选
        }
    """
    if not self.is_loaded:
        return {'success': False, 'message': '模型未加载'}
    
    try:
        start_time = time.time()
        
        # 1. 加载图像
        image = Image.open(image_path)
        
        # 2. 预处理
        input_tensor = self.preprocess(image)
        
        # 3. 推理
        output_tensor = self.model(input_tensor)
        
        # 4. 后处理
        output_image = self.postprocess(output_tensor)
        
        # 5. 保存
        output_image.save(output_path)
        
        inference_time = time.time() - start_time
        
        return {
            'success': True,
            'output_path': output_path,
            'inference_time': inference_time,
            'message': '推理成功',
            'metadata': {
                # 添加你想返回的信息
                'param1': value1,
            }
        }
    except Exception as e:
        return {'success': False, 'message': f'推理失败: {str(e)}'}
```

#### (4) `batch_inference()` - 批量推理

```python
def batch_inference(
    self,
    image_paths: List[str],
    output_dir: str,
    **kwargs
) -> Dict[str, Any]:
    """批量推理"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    succeeded = 0
    failed = 0
    start_time = time.time()
    
    for image_path in image_paths:
        # 生成输出路径
        output_path = output_dir / f"output_{Path(image_path).stem}.png"
        
        # 调用单张推理
        result = self.inference(image_path, str(output_path), **kwargs)
        
        results.append({
            'input_path': image_path,
            'output_path': str(output_path) if result['success'] else None,
            'success': result['success'],
            'inference_time': result.get('inference_time', 0),
            'message': result['message']
        })
        
        if result['success']:
            succeeded += 1
        else:
            failed += 1
    
    return {
        'success': True,
        'total': len(image_paths),
        'succeeded': succeeded,
        'failed': failed,
        'results': results,
        'total_time': time.time() - start_time,
        'message': f'批量推理完成: {succeeded}/{len(image_paths)}'
    }
```

#### (5) `get_model_info()` - 获取模型信息

```python
def get_model_info(self) -> Dict[str, Any]:
    """返回模型信息（用于前端展示）"""
    info = {
        'name': '你的模型名称',
        'version': '1.0',
        'description': '模型描述',
        'input_size': (512, 512),  # (H, W)
        'output_size': (512, 512),
        'supported_formats': ['.png', '.jpg', '.jpeg'],
        'default_params': {
            'ode_steps': 50,
            'param2': value2,
        },
        'custom_fields': {
            # 自定义字段
        }
    }
    
    # 如果模型已加载，添加参数量等信息
    if self.is_loaded and self.model is not None:
        info['parameters'] = sum(p.numel() for p in self.model.parameters())
    
    return info
```

---

## 📚 **完整示例：Flow Matching V2插件**

参考 `plugins/flow_matching_v2_plugin.py`：

```python
class FlowMatchingV2Plugin(InferenceInterface):
    def __init__(self, plugin_name: str, config: Dict[str, Any]):
        super().__init__(plugin_name, config)
        self.base_channels = config.get('base_channels', 64)
        self.image_size = config.get('image_size', (512, 512))
        
    def load_model(self, checkpoint_path: str, device: str = 'cuda:0') -> bool:
        # ... 实现加载逻辑
        
    def inference(self, image_path, output_path, **kwargs):
        # 1. 加载图像
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 2. ODE求解生成图像
        output_tensor = self._ode_solver(
            sim_image=image_tensor,
            ode_steps=kwargs.get('ode_steps', 50)
        )
        
        # 3. 保存
        output_image = self._tensor_to_image(output_tensor[0])
        output_image.save(output_path)
        
        # 4. 返回结果
        return {
            'success': True,
            'output_path': output_path,
            'inference_time': inference_time,
            'message': '推理成功'
        }
```

---

## 🔧 **高级功能**

### 1. 自定义配置验证

重写 `validate_config()` 方法：

```python
def validate_config(self) -> bool:
    """验证配置是否合法"""
    required_keys = ['checkpoint_path', 'device', 'your_param']
    for key in required_keys:
        if key not in self.config:
            print(f"缺少配置参数: {key}")
            return False
    return True
```

### 2. 性能指标跟踪

使用 `InferenceMetrics`：

```python
from webui.backend.core import InferenceMetrics

class MyPlugin(InferenceInterface):
    def __init__(self, plugin_name, config):
        super().__init__(plugin_name, config)
        self.metrics = InferenceMetrics()
    
    def inference(self, ...):
        # ... 推理逻辑 ...
        self.metrics.update(inference_time)
        
    def get_model_info(self):
        info = {...}
        if self.is_loaded:
            info['performance'] = self.metrics.get_stats()
        return info
```

### 3. 自定义预处理/后处理

重写可选方法：

```python
def preprocess(self, image: np.ndarray) -> torch.Tensor:
    """自定义预处理"""
    # 1. Resize
    image = cv2.resize(image, self.image_size)
    
    # 2. 归一化
    image = image / 255.0
    
    # 3. 转为Tensor
    tensor = torch.from_numpy(image).float()
    return tensor.unsqueeze(0).to(self.device)

def postprocess(self, output: torch.Tensor) -> np.ndarray:
    """自定义后处理"""
    # 1. 去归一化
    output = torch.clamp(output, 0, 1) * 255
    
    # 2. 转为numpy
    array = output.cpu().numpy().astype(np.uint8)
    
    return array
```

### 4. 接收前端自定义参数

前端传入的参数通过 `**kwargs` 接收：

```python
def inference(self, image_path, output_path, **kwargs):
    # 获取前端传入的参数
    temperature = kwargs.get('temperature', 1.0)
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    
    # 使用这些参数
    output = self.model(input, temperature=temperature, guidance_scale=guidance_scale)
```

前端在 `InferenceParams.vue` 中添加对应的输入框。

---

## 📦 **注册插件**

### 方式1：代码注册

在 `backend/main.py` 中：

```python
from plugins.my_model_plugin import MyModelPlugin

@app.on_event("startup")
async def startup_event():
    model_manager.register_plugin(
        plugin_name='my_model',
        plugin_class=MyModelPlugin,
        config={
            'checkpoint_path': '/path/to/checkpoint.pth',
            'device': 'cuda:0',
            'your_param': value
        }
    )
```

### 方式2：动态加载

```python
model_manager.load_plugin_from_file(
    plugin_file='/path/to/my_model_plugin.py',
    plugin_class_name='MyModelPlugin',
    plugin_name='my_model',
    config={...}
)
```

---

## ✅ **测试插件**

在插件文件末尾添加测试代码：

```python
if __name__ == "__main__":
    config = {
        'checkpoint_path': '/path/to/checkpoint.pth',
        'device': 'cuda:0',
    }
    
    plugin = MyModelPlugin('test', config)
    
    # 测试加载
    if plugin.load_model(config['checkpoint_path']):
        print("\n模型信息:")
        import json
        print(json.dumps(plugin.get_model_info(), indent=2))
        
        # 测试推理
        result = plugin.inference(
            image_path='/path/to/test.png',
            output_path='/tmp/output.png'
        )
        print(json.dumps(result, indent=2))
        
        plugin.unload_model()
```

运行测试：

```bash
cd webui/backend
python plugins/my_model_plugin.py
```

---

## 🐛 **常见问题**

### Q: 如何处理不同输入尺寸？

**A**: 在 `preprocess()` 中统一resize：

```python
def preprocess(self, image):
    if image.size != self.image_size:
        image = image.resize(self.image_size, Image.BILINEAR)
    return self.transform(image)
```

### Q: 如何支持批量推理加速？

**A**: 重写 `batch_inference()` 使用批量处理：

```python
def batch_inference(self, image_paths, output_dir, **kwargs):
    # 批量加载
    images = [self.preprocess(Image.open(p)) for p in image_paths]
    batch = torch.stack(images)
    
    # 批量推理
    with torch.no_grad():
        outputs = self.model(batch)
    
    # 批量保存
    for i, output_path in enumerate(output_paths):
        self.postprocess(outputs[i]).save(output_path)
```

### Q: 如何处理CUDA OOM？

**A**: 在 `inference()` 中添加异常处理：

```python
def inference(self, ...):
    try:
        output = self.model(input)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return {
                'success': False,
                'message': 'CUDA内存不足，请降低batch_size或图像尺寸'
            }
        raise e
```

---

## 📋 **检查清单**

开发完成后，确认以下事项：

- [ ] 继承自 `InferenceInterface`
- [ ] 实现所有5个抽象方法
- [ ] `load_model()` 设置 `self.is_loaded = True`
- [ ] `inference()` 返回正确格式的字典
- [ ] `get_model_info()` 返回完整信息
- [ ] 通过单元测试
- [ ] 在WebUI中测试推理
- [ ] 添加文档注释

---

## 🎉 **完成！**

现在你的插件已经可以在WebUI中使用了！

如有问题，请参考 `plugins/flow_matching_v2_plugin.py` 示例。

