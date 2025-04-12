"""Attempts at regression compatible layers"""
from qiskit import QuantumCircuit
from quantum_launcher.routines.qiskit_routines import QiskitBackend
import torch

from qailab.torch.qlayer import QLayer
from qailab.torch.autograd import ArgMax


class ExpectedValueQLayer(QLayer):
    """
    This layer returns the expected value of a bitstring sampled from the underlying quantum circuit.
    The value is a floating point number in range <0, 2^num_measured_qubits - 1>
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: QiskitBackend | None = None,
        shots: int = 1024,
        rescale_output: tuple[float, float] | None = None
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
        super().__init__(circuit, backend=backend, shots=shots)
        self._max_expected_out_value = self.out_features - 1
        self.out_features = 1
        self._rescale_output_range = rescale_output

    def _rescale_out(self, x: torch.Tensor):
        if self._rescale_output_range is None:
            return x

        out_min, out_max = self._rescale_output_range
        x_zero_one = x / max(self._max_expected_out_value, 1)
        x_scale = x_zero_one * (out_max - out_min) + out_min
        return x_scale

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out_distribution = super().forward(input_tensor)

        def make_vals(l):
            return torch.tensor(
                list(range(l)),
                dtype=out_distribution.dtype,
                requires_grad=out_distribution.requires_grad
            )

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

    def __init__(self, circuit: QuantumCircuit, *, backend: QiskitBackend | None = None, shots: int = 1024) -> None:
        super().__init__(circuit, backend=backend, shots=shots)
        self.out_features = 1

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out_distribution = super().forward(input_tensor)
        argmax = ArgMax.apply(out_distribution)
        if not isinstance(argmax, torch.Tensor):
            raise ValueError(f"Argmax output error. {type(argmax)} is not torch.Tensor")
        return argmax
