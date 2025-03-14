""" Forward pass algorithm implementation in quantum_launcher. """
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from quantum_launcher.base import Algorithm
from quantum_launcher.base.base import Backend, Problem, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend


class ForwardPass(Algorithm):
    """Forward Pass"""
    _algorithm_format = 'none'

    def __init__(self, shots: int = 1024) -> None:
        """_summary_
        """
        self.shots = shots
        super().__init__()

    def run(self, problem: Problem, backend: Backend, formatter: Callable[..., Any] | None = None) -> Result:
        if formatter is None:
            raise ValueError('Formatter for Forward pass not found!')
        if not isinstance(backend, QiskitBackend):
            raise ValueError('Wrong sampler given into')
        pubs = formatter(problem)
        sampler = backend.sampler
        job = sampler.run(pubs, shots=self.shots)
        result = job.result()
        data = result._pub_results[0].data['c'].array  # pylint: disable=protected-access
        distribution = defaultdict(float)
        for i in data:
            distribution[tuple(i)] += 1/self.shots
        return Result('', 0, '', 0, distribution, {}, self.shots, 0, 0, data)
