""" QLauncher problem implementation """
from typing import Any
from quantum_launcher.base import Problem


class CircuitProblem(Problem):
    def __init__(self, instance: Any, instance_name: str = 'unnamed') -> None:
        ...
        super().__init__(instance, instance_name)
