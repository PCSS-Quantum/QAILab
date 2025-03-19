""" Module with QLayer """
import math
from torch import Tensor, nn
import torch
from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qiskit import QuantumCircuit


from qlearning.qlauncher import CircuitProblem, ForwardPass, BackwardPass


class QLayer(nn.Module):
    """ Base quantum layer class """
    weight: Tensor

    def __init__(
        self,
        circuit: QuantumCircuit
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((len(circuit.parameters), 1))
        )
        self.reset_parameters()
        self.circuit = circuit
        self.circuit_pr = CircuitProblem(self.circuit)
        self.launcher_forward = QuantumLauncher(self.circuit_pr, ForwardPass(), QiskitBackend('local_simulator'))
        self.launcher_backward = QuantumLauncher(self.circuit_pr, BackwardPass('parameter-shift'))

    def reset_parameters(self) -> None:
        """ Parameter reset """
        nn.init.uniform_(self.weight, 0, 2 * math.pi)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """ Forward """
        weight = self.weight.detach().cpu().numpy()  # pylint: disable=not-callable
        input_array = input_tensor.detach().cpu().numpy()
        result = self.launcher_forward.run(initial_state=input_array, parameters=weight[0])
        output_array = self._postprocess(result)
        return Tensor(output_array)

    def extra_repr(self) -> str:
        return f"{self.circuit}"

    def _postprocess(self, result: Result):
        return max(result.distribution, key=lambda x: result.distribution[x])
