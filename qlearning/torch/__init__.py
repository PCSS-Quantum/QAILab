""" Torch plugin for QLearning """
from .qlayer import QLayer, ExpQLayer
from .qmodel import QModel

__all__ = ["QLayer", "QModel", "ExpQLayer"]
