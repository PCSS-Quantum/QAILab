"""Forward pass algorithm implementation in qlauncher."""

from qlauncher.base import Algorithm
from qlauncher.base.base import Result
from qlauncher.routines.qiskit import QiskitBackend

from qailab.qlauncher.problem import ParameterizedCircuit
from qailab.utils import bitstring_to_bit_tuple


class ForwardPass(Algorithm[ParameterizedCircuit, QiskitBackend]):
	"""Forward Pass"""

	_algorithm_format = 'none'

	def __init__(self, shots: int = 1024) -> None:
		"""Forward pass implementation for QLauncher.

		Args:
				shots (int): Number of shots. Defaults to 1024.
		"""
		self.shots = shots
		super().__init__()

	def _run(self, problem: ParameterizedCircuit, backend: QiskitBackend) -> Result:
		distribution = backend.sample_circuit(problem.bound_circuit, shots=self.shots)
		sum_counts = sum(distribution.values())
		distribution = {bitstring_to_bit_tuple(k): v / sum_counts for k, v in distribution.items()}

		return Result('', 0, '', 0, distribution, {}, self.shots, 0, 0, None)  # Results are not picklable

	def run(self, problem: ParameterizedCircuit, backend: QiskitBackend) -> Result:
		# TODO(tbd): multiprocessed layers sometimes freeze execution on KeyboardInterrupt. This helps somewhat but it still happens
		try:
			return self._run(problem, backend)
		except KeyboardInterrupt:
			return Result.from_counts_energies({}, {})
