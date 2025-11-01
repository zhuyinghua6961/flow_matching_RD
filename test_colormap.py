"""
测试Colormap功能
快速验证不同colormap的效果
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'webui' / 'backend'))

from plugins.flow_matching_v2_plugin import FlowMatchingV2Plugin

def test_colormap():
    print("=" * 70)
    print("           Colormap功能测试")
    print("=" * 70)
    
    # 创建插件实例
    config = {
        'checkpoint_path': 'outputs_v2/checkpoints/checkpoint_epoch_31.pth',
        'device': 'cuda:0',
        'base_channels': 64,
        'channel_mult': (1, 2, 4, 8),
        'attention_levels': (),
        'apply_colormap': True,
        'colormap_name': 'jet'
    }
    
    plugin = FlowMatchingV2Plugin('flow_matching_v2', config)
    
    # 加载模型
    print("\n加载模型...")
    success = plugin.load_model(
        checkpoint_path=config['checkpoint_path'],
        device=config['device']
    )
    
    if not success:
        print("❌ 模型加载失败")
        return
    
    # 测试图像
    test_image = 'dataset/test/sim/rd1601.png'
    
    # 测试不同colormap
    colormaps_to_test = ['jet', 'viridis', 'plasma', 'hot', 'gray']
    
    print(f"\n测试图像: {test_image}")
    print(f"将生成 {len(colormaps_to_test)} 种colormap效果\n")
    
    output_dir = Path('webui/backend/outputs_colormap_test')
    output_dir.mkdir(exist_ok=True)
    
    for i, cmap in enumerate(colormaps_to_test, 1):
        print(f"[{i}/{len(colormaps_to_test)}] 测试 {cmap} colormap...")
        
        output_path = output_dir / f"output_{cmap}.png"
        
        result = plugin.inference(
            image_path=test_image,
            output_path=str(output_path),
            ode_steps=50,
            colormap=cmap if cmap != 'gray' else None,
            apply_colormap=(cmap != 'gray')
        )
        
        if result['success']:
            print(f"  ✅ 成功: {output_path}")
            print(f"  ⏱️  耗时: {result['inference_time']:.2f}s")
        else:
            print(f"  ❌ 失败: {result['message']}")
    
    print(f"\n" + "=" * 70)
    print(f"测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"")
    print(f"请对比不同colormap的效果:")
    for cmap in colormaps_to_test:
        print(f"  - output_{cmap}.png")
    print("=" * 70)
    
    # 清理
    plugin.unload_model()

if __name__ == "__main__":
    test_colormap()
