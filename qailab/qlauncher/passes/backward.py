"""Backward pass algorithm implementation in qlauncher."""

from typing import Literal

from qlauncher.base.base import Result
from qlauncher.routines.qiskit import QiskitBackend

from qailab.gradient.gradient_calculation import calculate_jacobian
from qailab.qlauncher.passes.forward import ForwardPass
from qailab.qlauncher.problem import ParameterizedCircuit


class BackwardPass(ForwardPass):
    """Backward Pass, calculates two jacobian matrices: w.r.t. to input and w.r.t. weights"""

    def __init__(self, gradient_method: Literal["param_shift", "spsa", "lin_comb"] = "param_shift", shots: int = 1024) -> None:
        self.gradient_method = gradient_method
        self.shots = shots
        super().__init__()

    def _run(self, problem: ParameterizedCircuit, backend: QiskitBackend) -> Result:
        circuit, params = problem._circuit, problem._parameter_values

        input_params = {x: params[x] for x in params if x.name.startswith("input")}
        weight_params = {x: params[x] for x in params if x.name.startswith("weight")}

        # All other params must be assigned.
        input_jacobian = calculate_jacobian(
            circuit.assign_parameters(weight_params), input_params, backend, self.gradient_method, self.shots
        )
        weight_jacobian = calculate_jacobian(
            circuit.assign_parameters(input_params), weight_params, backend, self.gradient_method, self.shots
        )
        return Result("", 0, "", 0, {}, {}, self.shots, 0, 0, {"input": input_jacobian, "weight": weight_jacobian})

    def run(self, problem: ParameterizedCircuit, backend: QiskitBackend) -> Result:
        # TODO multiprocessed layers sometimes freeze execution on KeyboardInterrupt. This helps somewhat but it still happens
        try:
            return self._run(problem, backend)
        except KeyboardInterrupt:
            return None
