"""Derivative calculation for parameterized quantum circuits."""
from typing import Literal

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit import Parameter
from qiskit_algorithms.gradients import (
    BaseSamplerGradient,
    LinCombSamplerGradient,
    SPSASamplerGradient,
    FiniteDiffSamplerGradient,
    ParamShiftSamplerGradient,
    # SamplerGradientResult
)

from quantum_launcher.base.base import Backend
from quantum_launcher.routines.qiskit_routines import QiskitBackend


def _param_grads_to_jacobian(grads, num_possible_values) -> np.ndarray:
    grads_list = [[g.get(i, 0) for i in range(num_possible_values)] for g in grads]
    return np.array(grads_list)


def calculate_jacobian(
    circuit: QuantumCircuit,
    set_params: dict[Parameter, int | float],
    backend: Backend,
    method: Literal['param_shift', 'spsa', 'lin_comb', 'fin_diff'] = 'param_shift',
    shots: int = 1024
) -> np.ndarray:
    """
    For each parameter calculate partial derivatives w.r.t to each output value (possible measurement).

    Args:
        circuit (QuantumCircuit): Circuit to sample.
        set_params (dict[Parameter, int  |  float]): Parameters for which to calculate derivatives and their current values.
        backend (Backend): Backend to use.
        method (Literal[&#39;param_shift&#39;, &#39;spsa&#39;, &#39;lin_comb&#39;, &#39;fin_diff&#39;], optional):
        Gradient algorithm to use. Defaults to 'param_shift'.
        shots (int): How many shots to use for Sampler.

    Raises:
        ValueError: For unsupported method or backend.

    Returns:
        np.ndarray: `len(set_params) x 2^measured_qubits` matrix of partial derivatives.
    """
    gradient_type = {
        'param_shift': ParamShiftSamplerGradient,
        'spsa': lambda sampler: SPSASamplerGradient(sampler, epsilon=0.001, batch_size=10),
        'lin_comb': LinCombSamplerGradient,
        'fin_diff': lambda sampler: FiniteDiffSamplerGradient(sampler, epsilon=0.001)
    }.get(method, None)

    if gradient_type is None:
        raise ValueError(f"Unsupported method {method}")

    if not isinstance(backend, QiskitBackend):
        raise ValueError("Only qiskit backends are supported.")

    num_possible_values = 2**circuit.num_clbits

    #! TODO: change this sampler to backend sampler when adapter is fixed!!!!
    gradient_calc: BaseSamplerGradient = gradient_type(Sampler())

    params, values = zip(*list(set_params.items()))

    grads = gradient_calc.run([circuit], [values], [params], shots=shots).result().gradients[0]
    return _param_grads_to_jacobian(grads, num_possible_values)
