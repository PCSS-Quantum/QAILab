""" Backward pass algorithm implementation in quantum_launcher. """
from collections.abc import Callable
from typing import Any
from quantum_launcher.base.base import Backend, Problem, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qlearning.qlauncher.passes.forward import ForwardPass


class BackwardPass(ForwardPass):
    """ Backward Pass """

    def __init__(self, gradient_method: str = '', shots: int = 1024) -> None:
        self.gradient_method = gradient_method
        self.shots = shots
        super().__init__()

    def run(self, problem: Problem, backend: Backend, formatter: Callable[..., Any] | None = None) -> Result:
        if formatter is None:
            raise ValueError('Formatter for Backward pass not found!')
        if not isinstance(backend, QiskitBackend):
            raise ValueError('Wrong sampler given into')
        super().run(problem, backend)

        return Result('', 0, '', 0, {}, {}, self.shots, 0, 0, '')
