"""Attempts at regression compatible layers"""

import torch
from qiskit import QuantumCircuit
from qlauncher.routines.qiskit import QiskitBackend

from qailab.torch.autograd import ArgMax
from qailab.torch.qlayer import QLayer


class ExpectedValueQLayer(QLayer):
	"""
	This layer returns the expected value of a bitstring sampled from the underlying quantum circuit.
	The value is a floating point number in range <0, 2^num_measured_qubits - 1>
	"""

	def __init__(
		self,
		circuit: QuantumCircuit,
		*,
		backends: QiskitBackend | list[QiskitBackend] | None = None,
		shots: int = 1024,
		rescale_output: tuple[float, float] | None = None,
	) -> None:
		"""
		Args:
			circuit (QuantumCircuit): Circuit to run.
			backend (QiskitBackend | None, optional): Backend to use. If None a local QiskitBackend is created. Defaults to None.
			shots (int, optional): Number of times to sample the circuit. Defaults to 1024.
			rescale_output (tuple[float,float] | None, optional):
			Tuple of (low, high) representing a range to rescale output to.
			If None the output will be in range <0, 2^num_measured_qubits - 1>. Defaults to None.
		"""
		super().__init__(circuit, backends=backends, shots=shots)
		self._max_expected_out_value = self.out_features - 1
		self.out_features = 1
		self._rescale_output_range = rescale_output

	def _rescale_out(self, x: torch.Tensor) -> torch.Tensor:
		if self._rescale_output_range is None:
			return x

		out_min, out_max = self._rescale_output_range
		x_zero_one = x / max(self._max_expected_out_value, 1)
		return x_zero_one * (out_max - out_min) + out_min

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		out_distribution = super().forward(input_tensor)

		def make_vals(range_max: int) -> torch.Tensor:
			return torch.tensor(list(range(range_max)), dtype=out_distribution.dtype, requires_grad=out_distribution.requires_grad)

		# Unbatched input
		if len(out_distribution.shape) == 1:
			values = make_vals(out_distribution.shape[0])
			return torch.sum(out_distribution * values)
		# Batched input
		values = torch.stack([make_vals(out_distribution.shape[1])] * out_distribution.shape[0])
		summed = torch.sum(out_distribution * values, dim=-1)
		return self._rescale_out(summed)


class ArgmaxQLayer(QLayer):
	"""
	This layer returns the most common bitstring sampled from the underlying quantum circuit.
	The value is a whole number in range <0, 2^num_measured_qubits - 1>
	"""

	def __init__(self, circuit: QuantumCircuit, *, backends: QiskitBackend | list[QiskitBackend] | None = None, shots: int = 1024) -> None:
		super().__init__(circuit, backends=backends, shots=shots)
		self.out_features = 1

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		out_distribution = super().forward(input_tensor)
		argmax = ArgMax.apply(out_distribution)
		if not isinstance(argmax, torch.Tensor):
			raise ValueError(f'Argmax output error. {type(argmax)} is not torch.Tensor')
		return argmax
