#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热力图批量生成工具

用法:
    python generate_heatmaps.py
    python generate_heatmaps.py --config my_config.yaml
    python generate_heatmaps.py --prompt_file prompts.txt
"""

import argparse
import yaml
import re
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.heatmap import HeatmapGenerator


def parse_targets_prompt(prompt):
    """
    解析新格式的prompt
    
    格式: targets=[(距离1,速度1),(距离2,速度2),...]
    例如: targets=[(25,30),(30,-20)]
    
    Returns:
        list of dict: [{'distance': 25, 'speed': 30}, {'distance': 30, 'speed': -20}]
    """
    prompt = prompt.strip()
    
    # 使用正则提取所有 (数字,数字) 对
    pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
    matches = re.findall(pattern, prompt)
    
    if not matches:
        raise ValueError(f"无法解析prompt: {prompt}")
    
    targets = []
    for distance_str, speed_str in matches:
        targets.append({
            'distance': float(distance_str),
            'speed': float(speed_str)
        })
    
    return targets


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts_from_file(file_path):
    """从txt文件加载prompts"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                prompts.append(line)
    return prompts


def save_grayscale_heatmap(heatmap, save_path):
    """
    保存灰度热力图
    
    Args:
        heatmap: (H, W) numpy array, 范围[0, 1]
        save_path: 保存路径
    """
    # 转换为0-255的灰度图
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # 保存图片
    img = Image.fromarray(heatmap_uint8)
    img.save(save_path)


def save_info_file(info_path, prompt, targets, heatmap, pixel_coords):
    """保存热力图信息到txt文件"""
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("热力图信息\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"目标数量: {len(targets)}\n\n")
        
        for i, (target, (x, y)) in enumerate(zip(targets, pixel_coords), 1):
            f.write(f"目标 {i}:\n")
            f.write(f"  距离: {target['distance']:.2f} m\n")
            f.write(f"  速度: {target['speed']:+.2f} m/s\n")
            f.write(f"  像素坐标: X={x:.1f}, Y={y:.1f}\n\n")
        
        f.write(f"热力图统计:\n")
        f.write(f"  形状: {heatmap.shape}\n")
        f.write(f"  最小值: {heatmap.min():.6f}\n")
        f.write(f"  最大值: {heatmap.max():.6f}\n")
        f.write(f"  均值: {heatmap.mean():.6f}\n")
        f.write(f"  非零像素: {np.sum(heatmap > 0.01)}\n")
        f.write(f"  峰值像素 (>0.9): {np.sum(heatmap > 0.9)}\n")


def generate_heatmap_from_targets(generator, targets):
    """
    从目标列表生成热力图
    
    Args:
        generator: HeatmapGenerator实例
        targets: list of dict [{'distance': float, 'speed': float}, ...]
    
    Returns:
        heatmap: (H, W) numpy array
        pixel_coords: list of (x, y) tuples
    """
    # 转换物理坐标到像素坐标
    pixel_coords = []
    for target in targets:
        x, y = generator.params_to_pixel(target['speed'], target['distance'])
        pixel_coords.append((x, y))
    
    # 生成热力图
    if len(pixel_coords) == 1:
        heatmap = generator.generate_gaussian_heatmap(pixel_coords[0])
    else:
        heatmap = generator.generate_multi_target_heatmap(pixel_coords)
    
    return heatmap, pixel_coords


def main():
    parser = argparse.ArgumentParser(description='热力图批量生成工具')
    parser.add_argument('--config', type=str, default='heatmap_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='prompt文件路径（覆盖配置文件中的设置）')
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("请先创建配置文件或使用默认配置文件: heatmap_config.yaml")
        return
    
    config = load_config(args.config)
    
    # 创建热力图生成器
    generator = HeatmapGenerator(
        img_size=config['img_size'],
        max_speed=config['max_speed'],
        max_range=config['max_range'],
        sigma=config['sigma']
    )
    
    # 创建输出目录
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取prompts
    if args.prompt_file:
        # 从命令行指定的文件读取
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"✓ 从文件加载了 {len(prompts)} 个prompts: {args.prompt_file}")
    elif config['input']['prompt_file']:
        # 从配置文件指定的文件读取
        prompts = load_prompts_from_file(config['input']['prompt_file'])
        print(f"✓ 从文件加载了 {len(prompts)} 个prompts: {config['input']['prompt_file']}")
    else:
        # 从配置文件中读取
        prompts = config['input']['prompts']
        print(f"✓ 从配置文件加载了 {len(prompts)} 个prompts")
    
    if not prompts:
        print("❌ 没有找到任何prompt")
        return
    
    # 打印配置
    print("\n" + "="*70)
    print("热力图生成配置")
    print("="*70)
    print(f"  图像尺寸: {config['img_size']}×{config['img_size']}")
    print(f"  速度范围: ±{config['max_speed']} m/s")
    print(f"  距离范围: 0-{config['max_range']} m")
    print(f"  高斯标准差: {config['sigma']}")
    print(f"  输出目录: {output_dir}")
    print(f"  只输出灰度图: {config['output']['grayscale_only']}")
    print(f"  保存信息文件: {config['output']['save_info']}")
    print("="*70 + "\n")
    
    # 生成热力图
    prefix = config['output']['prefix']
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] 处理: {prompt}")
        
        try:
            # 解析prompt
            targets = parse_targets_prompt(prompt)
            print(f"  → 目标数量: {len(targets)}")
            
            # 生成热力图
            heatmap, pixel_coords = generate_heatmap_from_targets(generator, targets)
            
            # 保存热力图
            filename = f"{prefix}_{idx:03d}.png"
            save_path = output_dir / filename
            save_grayscale_heatmap(heatmap, save_path)
            print(f"  ✓ 已保存: {save_path}")
            
            # 保存信息文件
            if config['output']['save_info']:
                info_path = output_dir / f"{prefix}_{idx:03d}_info.txt"
                save_info_file(info_path, prompt, targets, heatmap, pixel_coords)
                print(f"  ✓ 信息文件: {info_path}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue
        
        print()
    
    print("="*70)
    print(f"✓ 完成！共生成 {len(prompts)} 个热力图")
    print(f"✓ 输出目录: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()

