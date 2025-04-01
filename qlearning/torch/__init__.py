""" Torch plugin for QLearning """
from .qlayer import QLayer
from .qmodel import QModel
from .layers.regression import ExpectedValueQLayer, ArgmaxQLayer

__all__ = ['QLayer', 'QModel', 'ExpectedValueQLayer', 'ArgmaxQLayer']
