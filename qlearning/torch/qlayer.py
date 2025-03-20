""" Module with QLayer """
import math
from torch import Tensor, nn
import torch
from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qiskit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.isometry import Isometry

from qlearning.qlauncher import CircuitProblem, ForwardPass, BackwardPass
Isometry.__init__.__defaults__ = (1e-6,)  # FIXME: If anyone has any idea, feel free


class QLayer(nn.Module):
    """ Base quantum layer class """
    weight: Tensor

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.trainable_params = list(filter(lambda x: not x.name.startswith('input'), circuit.parameters))
        self.weight = nn.Parameter(
            torch.empty((len(self.trainable_params), 1))
        )
        self._input_parameters = list(filter(lambda x: x.name.startswith('input'), circuit.parameters)) or None
        self.reset_parameters()
        self.circuit = circuit
        self.circuit_pr = CircuitProblem(self.circuit)
        self.launcher_forward = QuantumLauncher(self.circuit_pr, ForwardPass(shots=shots), QiskitBackend('local_simulator'))
        self.launcher_backward = QuantumLauncher(self.circuit_pr, BackwardPass('parameter-shift', shots=shots))

    def reset_parameters(self) -> None:
        """ Parameter reset """
        nn.init.uniform_(self.weight, 0, 2 * math.pi)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """ Forward """
        weight = self.weight.detach().cpu().numpy()  # pylint: disable=not-callable
        parameters = dict(zip(self.trainable_params, weight[0]))
        input_array = input_tensor.detach().cpu().numpy()
        if self._input_parameters is not None:
            input_params = dict(zip(self._input_parameters, input_array))
            parameters.update(input_params)
            input_array = None
        result = self.launcher_forward.run(initial_state=input_array, parameters=parameters)
        output_array = self._postprocess(result)
        return Tensor(output_array)

    def extra_repr(self) -> str:
        return f"{self.circuit}"

    def _postprocess(self, result: Result):
        return max(result.distribution, key=lambda x: result.distribution[x])
