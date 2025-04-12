""" Backward pass algorithm implementation in quantum_launcher. """
from collections.abc import Callable
from typing import Any, Literal
from quantum_launcher.base.base import Backend, Problem, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qailab.qlauncher.passes.forward import ForwardPass
from qailab.gradient.gradient_calculation import calculate_jacobian


class BackwardPass(ForwardPass):
    """ Backward Pass, calculates two jacobian matrices: w.r.t. to input and w.r.t. weights"""

    def __init__(self, gradient_method: Literal['param_shift', 'spsa', 'lin_comb'] = 'param_shift', shots: int = 1024) -> None:
        self.gradient_method = gradient_method
        self.shots = shots
        super().__init__()

    def run(self, problem: Problem, backend: Backend, formatter: Callable[..., Any] | None = None) -> Result:
        if formatter is None:
            raise ValueError('Formatter for Backward pass not found!')
        if not isinstance(backend, QiskitBackend):
            raise ValueError('Wrong sampler given into')

        format_result = formatter(problem)[0]
        if not isinstance(format_result, tuple):
            raise ValueError("Don't use autobind")

        circuit, params = format_result

        input_params = {x: params[x] for x in params if x.name.startswith('input')}
        weight_params = {x: params[x] for x in params if x.name.startswith('weight')}

        # All other params must be assigned.
        input_jacobian = calculate_jacobian(
            circuit.assign_parameters(weight_params),
            input_params,
            backend,
            self.gradient_method,
            self.shots
        )
        weight_jacobian = calculate_jacobian(
            circuit.assign_parameters(input_params),
            weight_params,
            backend,
            self.gradient_method,
            self.shots
        )

        return Result('', 0, '', 0, {}, {}, self.shots, 0, 0, {'input': input_jacobian, 'weight': weight_jacobian})
