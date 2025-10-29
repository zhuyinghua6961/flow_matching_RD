"""
早停机制 - Early Stopping
当验证损失在N个epoch内没有改善时停止训练
"""
import numpy as np


class EarlyStopping:
    """
    早停回调
    
    当监控指标在patience个epoch内没有改善min_delta时，停止训练
    """
    
    def __init__(
        self,
        patience=20,
        min_delta=0.0001,
        mode='min',
        verbose=True
    ):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善阈值
            mode: 'min' (损失) 或 'max' (准确率等)
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # 根据模式设置比较函数
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = np.inf
        else:
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -np.inf
    
    def __call__(self, current_score, epoch):
        """
        检查是否应该早停
        
        Args:
            current_score: 当前epoch的监控指标值
            epoch: 当前epoch编号
        
        Returns:
            should_stop: 是否应该停止训练
        """
        if self.best_score is None or (self.mode == 'min' and current_score == np.inf):
            # 第一次调用
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        if self.is_better(current_score, self.best_score):
            # 有改善
            if self.verbose:
                improvement = abs(current_score - self.best_score)
                print(f"  ✓ {self.mode=='min' and '损失' or '指标'}改善: "
                      f"{self.best_score:.6f} → {current_score:.6f} "
                      f"(改善 {improvement:.6f})")
            
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 无改善
            self.counter += 1
            
            if self.verbose:
                print(f"  ✗ 无改善 ({self.counter}/{self.patience}): "
                      f"当前 {current_score:.6f} vs 最佳 {self.best_score:.6f} "
                      f"(epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠️  早停触发！")
                    print(f"  最佳{self.mode=='min' and '损失' or '指标'}: {self.best_score:.6f}")
                    print(f"  最佳epoch: {self.best_epoch}")
                    print(f"  已等待: {self.counter} epochs 无改善")
                return True
        
        return False
    
    def state_dict(self):
        """保存状态（用于checkpoint）"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode
        }
    
    def load_state_dict(self, state_dict):
        """加载状态（从checkpoint恢复）"""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.patience = state_dict.get('patience', self.patience)
        self.min_delta = state_dict.get('min_delta', self.min_delta)
        self.mode = state_dict.get('mode', self.mode)


if __name__ == "__main__":
    # 测试早停机制
    print("测试早停机制\n")
    
    # 模拟训练过程
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')
    
    # 模拟损失序列：下降 → 平稳 → 上升
    losses = [1.0, 0.8, 0.6, 0.5, 0.48, 0.47, 0.469, 0.468, 0.467, 0.468, 0.469, 0.47, 0.48, 0.5]
    
    for epoch, loss in enumerate(losses):
        print(f"\nEpoch {epoch}: loss = {loss:.3f}")
        should_stop = early_stopping(loss, epoch)
        
        if should_stop:
            print(f"\n训练在epoch {epoch}停止")
            break
    
    print(f"\n最终状态:")
    print(f"  最佳损失: {early_stopping.best_score:.6f}")
    print(f"  最佳epoch: {early_stopping.best_epoch}")

