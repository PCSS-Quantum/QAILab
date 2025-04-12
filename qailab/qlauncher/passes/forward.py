""" Forward pass algorithm implementation in quantum_launcher. """
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from quantum_launcher.base import Algorithm
from quantum_launcher.base.base import Backend, Problem, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.base.sampler_result import SamplerResult

from qailab.utils import number_to_bit_tuple


class ForwardPass(Algorithm):
    """Forward Pass"""
    _algorithm_format = 'none'

    def __init__(self, shots: int = 1024) -> None:
        """Forward pass implementation for QLauncher.

        Args:
            shots (int): Number of shots. Defaults to 1024.
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
        if isinstance(result, PrimitiveResult):
            distribution = self._extract_results_v2(result)[0]
        elif isinstance(result, SamplerResult):
            distribution = self._extract_results_v1(result)[0]
        else:
            raise ValueError(f'Result with type: {type(result)} is not supported')
        return Result('', 0, '', 0, distribution, {}, self.shots, 0, 0, None)  # Results are not picklable

    def _extract_results_v2(self, result: PrimitiveResult) -> list[dict]:
        distributions = []
        for pub in result._pub_results:  # pylint: disable=protected-access
            data = pub.data['c'].array
            num_qubits = pub.data['c'].num_bits
            distribution = defaultdict(float)
            for datum_arr in data:
                # Qiskit splits measurements into 8 bit chunks for some godforsaken reason.
                tot_num = 0
                for i, v in enumerate(datum_arr[::-1]):
                    tot_num += int(v) * (2**(i * 8))

                distribution[number_to_bit_tuple(tot_num, num_qubits)] += 1 / self.shots
            distributions.append(distribution)
        return distributions

    def _extract_results_v1(self, result: SamplerResult) -> list[dict]:
        distributions = []
        for quasi_dist in result.quasi_dists:
            distribution = {}
            for key, value in quasi_dist.items():
                distribution[number_to_bit_tuple(key, quasi_dist._num_bits)] = value  # pylint: disable=protected-access
            distributions.append(distribution)
        return distributions
