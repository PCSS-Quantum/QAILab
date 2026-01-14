""" Torch plugin for QAILab """
from .layers.regression import ArgmaxQLayer, ExpectedValueQLayer
from .qlayer import QLayer
from .qmodel import QModel

__all__ = ['QLayer', 'QModel', 'ExpectedValueQLayer', 'ArgmaxQLayer']
