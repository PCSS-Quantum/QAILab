""" Module with QLayer """
import math

import torch
from torch import Tensor, nn

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library.generalized_gates.isometry import Isometry

from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qailab.circuit.utils import filter_params
from qailab.qlauncher import CircuitProblem, ForwardPass, BackwardPass
from qailab.torch.autograd import ExpVQCFunction
Isometry.__init__.__defaults__ = (1e-6,)  # FIXME: If anyone has any idea, feel free


class QLayer(nn.Module):
    """
    Base quantum layer class.

    This layer will output 2^num_measured_qubits features,
    which represent the distribution of measurements from the underlying quantum circuit.

    The bitstring order is [0, 1, 2, ..., 2^num_measured_qubits-1]
    """
    theta_trainable: Tensor

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: QiskitBackend | None = None,
        shots: int = 1024,
    ) -> None:
        super().__init__()

        self.theta_trainable = nn.Parameter(
            torch.empty((len(filter_params(circuit, 'weight')), 1))
        )

        self.reset_parameters()

        if backend is None:
            backend = QiskitBackend('local_simulator')

        self.circuit = transpile(circuit, backend.sampler.backend) if hasattr(backend.sampler, 'backend') else circuit
        self.circuit_pr = CircuitProblem(self.circuit)

        # Helper variables for hybrid networks
        self.in_features = len(filter_params(circuit, 'input'))
        self.out_features = 2**self.circuit.num_clbits

        self.launcher_forward = QuantumLauncher(
            self.circuit_pr,
            ForwardPass(shots=shots),
            backend
        )
        self.launcher_backward = QuantumLauncher(
            self.circuit_pr,
            BackwardPass('param_shift', shots=shots),
            backend
        )

    def reset_parameters(self) -> None:
        """ Parameter reset """
        nn.init.uniform_(self.theta_trainable, 0, 2 * math.pi)

    def extra_repr(self) -> str:
        return f"{self.circuit}"

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward run"""
        out = ExpVQCFunction.apply(
            input_tensor,
            self.theta_trainable[:, 0],
            self.launcher_forward,
            self.launcher_backward
        )
        if not isinstance(out, torch.Tensor):
            raise ValueError("Function did not return tensor output")
        return out
