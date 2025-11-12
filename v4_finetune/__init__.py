"""
V4版本 - 两阶段微调方案
基于预训练Flow Matching模型，用GAN专门优化多普勒效应
"""
from .discriminator_doppler import DopplerOnlyDiscriminator
from .finetune_trainer import FineTuneTrainer

__all__ = ['DopplerOnlyDiscriminator', 'FineTuneTrainer']
