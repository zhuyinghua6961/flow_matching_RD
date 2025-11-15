"""
Utils V2 - 纯图像对的工具函数
"""
from .dataset_v2 import RDPairDataset
from .losses import flow_matching_loss, frequency_domain_loss


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.0001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss if self.monitor == 'val_loss' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False


__all__ = [
    'RDPairDataset',
    'flow_matching_loss',
    'frequency_domain_loss',
    'EarlyStopping'
]

