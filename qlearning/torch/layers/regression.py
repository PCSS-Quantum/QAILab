"""Attempts at regression compatible layers"""
from qiskit import QuantumCircuit
from quantum_launcher.routines.qiskit_routines import QiskitBackend
import torch

from qlearning.torch.qlayer import QLayer
from qlearning.torch.autograd import ArgMax


class ExpectedValueQLayer(QLayer):
    """
    This layer returns the expected value of a bitstring sampled from the underlying quantum circuit.
    The value is a floating point number in range <0, 2^num_measured_qubits - 1>
    """

    def __init__(self, circuit: QuantumCircuit, *, backend: QiskitBackend | None = None, shots: int = 1024) -> None:
        super().__init__(circuit, backend=backend, shots=shots)
        self.out_features = 1

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
        return torch.sum(out_distribution * values, dim=-1)


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
