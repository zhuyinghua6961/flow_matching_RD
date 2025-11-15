#!/usr/bin/env python3
"""
测试V5 WGAN-GP的所有导入是否正常
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试V5 WGAN-GP导入...")
    
    try:
        # 测试基础导入
        print("  ✓ 测试基础库...")
        import torch
        import yaml
        import numpy as np
        from tqdm import tqdm
        
        # 测试utils_v2导入
        print("  ✓ 测试utils_v2...")
        from utils_v2 import RDPairDataset, frequency_domain_loss, EarlyStopping
        
        # 测试模型导入
        print("  ✓ 测试模型...")
        from models_v2.flow_matching_v2 import Sim2RealFlowModel
        
        # 测试Critic导入
        print("  ✓ 测试Critic...")
        from critic_doppler import DopplerOnlyCritic, doppler_wgan_gp_loss, doppler_feature_matching_loss
        
        # 测试训练器导入
        print("  ✓ 测试训练器...")
        from wgan_trainer import WGANTrainer
        
        print("\n🎉 所有导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 测试基本功能...")
    
    try:
        import torch
        # 测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  ✓ 设备: {device}")
        
        # 测试Critic创建
        from critic_doppler import DopplerOnlyCritic
        critic = DopplerOnlyCritic(base_channels=32)
        print(f"  ✓ Critic创建成功")
        
        # 测试EarlyStopping
        from utils_v2 import EarlyStopping
        early_stopping = EarlyStopping(patience=5)
        print(f"  ✓ EarlyStopping创建成功")
        
        # 测试配置加载
        config_path = "config_wgan.yaml"
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"  ✓ 配置文件加载成功")
        else:
            print(f"  ⚠️  配置文件不存在: {config_path}")
        
        print("\n🎉 基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("V5 WGAN-GP 导入和功能测试")
    print("="*60)
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        # 测试基本功能
        func_success = test_basic_functionality()
        
        if func_success:
            print(f"\n✅ 所有测试通过！可以开始训练了")
            print(f"\n🚀 启动训练命令:")
            print(f"   python train_wgan.py --config config_wgan.yaml")
        else:
            print(f"\n⚠️  导入成功但功能测试失败，请检查配置")
    else:
        print(f"\n❌ 导入失败，请先解决导入问题")
    
    print("="*60)
