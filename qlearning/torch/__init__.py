""" Torch plugin for QLearning """
from .qlayer import QLayer
from .layers.regression import ExpectedValueQLayer, ArgmaxQLayer

__all__ = ['QLayer', 'ExpectedValueQLayer', 'ArgmaxQLayer']
