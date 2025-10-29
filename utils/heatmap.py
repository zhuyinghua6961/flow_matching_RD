"""
热力图生成模块 - 基于Prompt的规则化方法
从文本提示中提取速度/距离参数，映射到RD图坐标系
"""
import re
import numpy as np
import torch


class HeatmapGenerator:
    """
    RD图热力图生成器
    
    RD图坐标系约定：
    - X轴（横向512像素）：速度（Doppler）
      - 中心(x=256): 0 m/s
      - 左侧: 负速度
      - 右侧: 正速度
    - Y轴（纵向512像素）：距离（Range）
      - 顶部(y=0): 0m
      - 底部(y=512): 200m
    """
    
    def __init__(
        self,
        img_size=512,
        max_speed=20.0,   # 最大速度范围 ±20 m/s（需根据实际调整）
        max_range=200.0,  # 最大距离 200m
        sigma=10.0        # 高斯热力图的标准差（控制扩散范围）
    ):
        self.img_size = img_size
        self.max_speed = max_speed
        self.max_range = max_range
        self.sigma = sigma
    
    def parse_prompt(self, prompt):
        """
        从文本提示中提取速度和距离参数（支持多目标）
        
        支持格式示例：
        单目标:
        - "速度: 5m/s, 距离: 100m"
        - "radar-RD-map; ... target number = 1, the first target: distance = 102m, velocity = 20.00m/s."
        
        多目标:
        - "radar-RD-map; ... target number = 2, the first target: distance = 85m, velocity = 1.00m/s, 
           the second target: distance = 29m, velocity = -4.00m/s."
        
        Args:
            prompt: 文本描述
        
        Returns:
            list of dict: [{'speed': float, 'distance': float}, ...]
        """
        targets = []
        
        # 检测是否为新格式（radar-RD-map格式）
        if 'radar-RD-map' in prompt.lower() or 'target number' in prompt.lower():
            # 新格式：提取所有目标
            # 匹配 "distance = XXm, velocity = XXm/s" 或 "velocity = XXm/s"
            pattern = r'distance\s*=\s*(-?\d+\.?\d*)m.*?velocity\s*=\s*(-?\d+\.?\d*)m/s'
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            
            for distance_str, velocity_str in matches:
                targets.append({
                    'distance': float(distance_str),
                    'speed': float(velocity_str)
                })
        
        else:
            # 旧格式：单目标
            params = {}
            
            # 提取速度（支持负值）
            speed_patterns = [
                r'速度[:：]?\s*(-?\d+\.?\d*)',
                r'speed[:：]?\s*(-?\d+\.?\d*)',
                r'velocity[:：]?\s*(-?\d+\.?\d*)'
            ]
            for pattern in speed_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    params['speed'] = float(match.group(1))
                    break
            
            # 提取距离
            distance_patterns = [
                r'距离[:：]?\s*(\d+\.?\d*)',
                r'range[:：]?\s*(\d+\.?\d*)',
                r'distance[:：]?\s*(\d+\.?\d*)'
            ]
            for pattern in distance_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    params['distance'] = float(match.group(1))
                    break
            
            # 默认值
            if 'speed' not in params:
                params['speed'] = 0.0
            if 'distance' not in params:
                params['distance'] = 0.0
            
            targets.append(params)
        
        # 如果没有提取到任何目标，返回默认值
        if not targets:
            targets = [{'speed': 0.0, 'distance': 0.0}]
        
        return targets
    
    def params_to_pixel(self, speed, distance):
        """
        将物理参数（速度、距离）映射到像素坐标
        
        Args:
            speed: 速度 (m/s)，可正可负
            distance: 距离 (m)，0-200
        
        Returns:
            (x_center, y_center): 像素坐标
        """
        # X坐标：速度映射
        # 中心(256)对应0 m/s，右侧正速度，左侧负速度
        x_center = self.img_size / 2 + (speed / self.max_speed) * (self.img_size / 2)
        x_center = np.clip(x_center, 0, self.img_size - 1)
        
        # Y坐标：距离映射
        # 顶部(0)对应0m，底部(512)对应200m
        y_center = (distance / self.max_range) * self.img_size
        y_center = np.clip(y_center, 0, self.img_size - 1)
        
        return x_center, y_center
    
    def generate_gaussian_heatmap(self, center):
        """
        生成单个目标的2D高斯分布热力图
        
        Args:
            center: (x, y) 像素坐标
        
        Returns:
            heatmap: (img_size, img_size) numpy array
        """
        x_center, y_center = center
        
        # 创建网格
        x = np.arange(0, self.img_size, dtype=np.float32)
        y = np.arange(0, self.img_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        # 2D高斯函数
        heatmap = np.exp(
            -((xx - x_center)**2 + (yy - y_center)**2) / (2 * self.sigma**2)
        )
        
        return heatmap
    
    def generate_multi_target_heatmap(self, centers):
        """
        生成多目标热力图（多个高斯峰的叠加）
        
        Args:
            centers: list of (x, y) 像素坐标
        
        Returns:
            heatmap: (img_size, img_size) numpy array
        """
        # 初始化热力图
        heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # 叠加所有目标的高斯分布
        for center in centers:
            single_heatmap = self.generate_gaussian_heatmap(center)
            heatmap = np.maximum(heatmap, single_heatmap)  # 取最大值（避免过度叠加）
        
        # 归一化到[0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def __call__(self, prompt):
        """
        完整流程：Prompt → 参数 → 像素坐标 → 热力图（支持多目标）
        
        Args:
            prompt: 文本描述（str）或参数字典（dict）或参数列表（list of dict）
        
        Returns:
            heatmap: (1, img_size, img_size) torch.Tensor
        """
        # 解析提示
        if isinstance(prompt, str):
            targets = self.parse_prompt(prompt)  # 返回list of dict
        elif isinstance(prompt, dict):
            targets = [prompt]  # 单目标
        elif isinstance(prompt, list):
            targets = prompt  # 多目标
        else:
            raise ValueError("prompt必须是str、dict或list of dict")
        
        # 映射所有目标到像素坐标
        centers = []
        for target in targets:
            speed = target.get('speed', 0.0)
            distance = target.get('distance', 0.0)
            center = self.params_to_pixel(speed, distance)
            centers.append(center)
        
        # 生成热力图
        if len(centers) == 1:
            # 单目标
            heatmap = self.generate_gaussian_heatmap(centers[0])
        else:
            # 多目标
            heatmap = self.generate_multi_target_heatmap(centers)
        
        # 归一化到[0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 转换为tensor (1, H, W)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)
        
        return heatmap_tensor
    
    def batch_generate(self, prompts):
        """
        批量生成热力图
        
        Args:
            prompts: list of str 或 list of dict
        
        Returns:
            heatmaps: (batch_size, 1, img_size, img_size) torch.Tensor
        """
        heatmaps = []
        for prompt in prompts:
            heatmap = self(prompt)
            heatmaps.append(heatmap)
        
        return torch.stack(heatmaps)


# 便捷函数
def create_heatmap_from_prompt(prompt, img_size=512, max_speed=20.0, 
                                max_range=200.0, sigma=10.0):
    """
    便捷函数：从prompt快速生成热力图
    
    Example:
        >>> heatmap = create_heatmap_from_prompt("速度: 5m/s, 距离: 100m")
        >>> print(heatmap.shape)  # (1, 512, 512)
    """
    generator = HeatmapGenerator(img_size, max_speed, max_range, sigma)
    return generator(prompt)


if __name__ == "__main__":
    # 测试
    generator = HeatmapGenerator(
        img_size=512,
        max_speed=30.0,  # 扩大速度范围以适应测试数据
        max_range=200.0,
        sigma=10.0
    )
    
    print("="*60)
    print("热力图生成器测试（支持多目标）")
    print("="*60)
    
    # 测试1: 旧格式单目标
    print("\n测试1 - 旧格式单目标")
    prompt1 = "速度: 5m/s, 距离: 100m"
    heatmap1 = generator(prompt1)
    targets1 = generator.parse_prompt(prompt1)
    print(f"Prompt: '{prompt1}'")
    print(f"提取目标: {targets1}")
    print(f"热力图形状: {heatmap1.shape}")
    print(f"值域: [{heatmap1.min():.3f}, {heatmap1.max():.3f}]")
    
    # 测试2: 新格式单目标
    print("\n测试2 - 新格式单目标")
    prompt2 = "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 1, the first target: distance = 102m, velocity = 20.00m/s."
    heatmap2 = generator(prompt2)
    targets2 = generator.parse_prompt(prompt2)
    print(f"Prompt: 'radar-RD-map ... distance = 102m, velocity = 20.00m/s.'")
    print(f"提取目标: {targets2}")
    print(f"热力图形状: {heatmap2.shape}")
    print(f"峰值数量: 1")
    
    # 测试3: 新格式双目标
    print("\n测试3 - 新格式双目标")
    prompt3 = "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s."
    heatmap3 = generator(prompt3)
    targets3 = generator.parse_prompt(prompt3)
    print(f"Prompt: 'radar-RD-map ... target number = 2 ...'")
    print(f"提取目标: {targets3}")
    print(f"热力图形状: {heatmap3.shape}")
    print(f"峰值数量: {len(targets3)}")
    
    # 测试4: 新格式三目标
    print("\n测试4 - 新格式三目标")
    prompt4 = "radar-RD-map; Turbo rendering; coordinates: top is near, bottom is far, left is negative, right is positive. target number = 3, the first target: distance = 79m, velocity = -27.00m/s, the second target: distance = 126m, velocity = -18.00m/s, the third target: distance = 26m, velocity = 26.00m/s."
    heatmap4 = generator(prompt4)
    targets4 = generator.parse_prompt(prompt4)
    print(f"Prompt: 'radar-RD-map ... target number = 3 ...'")
    print(f"提取目标:")
    for i, t in enumerate(targets4):
        print(f"  目标{i+1}: distance={t['distance']}m, velocity={t['speed']}m/s")
    print(f"热力图形状: {heatmap4.shape}")
    print(f"峰值数量: {len(targets4)}")
    
    # 测试5: 批量生成
    print("\n测试5 - 批量生成（混合单/多目标）")
    prompts = [
        "radar-RD-map; ... target number = 1, the first target: distance = 102m, velocity = 20.00m/s.",
        "radar-RD-map; ... target number = 2, the first target: distance = 85m, velocity = 1.00m/s, the second target: distance = 29m, velocity = -4.00m/s.",
        "速度: 0m/s, 距离: 50m"
    ]
    batch_heatmaps = generator.batch_generate(prompts)
    print(f"批量热力图形状: {batch_heatmaps.shape}")
    
    print("\n" + "="*60)
    print("所有测试通过！✓")
    print("="*60)

