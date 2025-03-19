"""Derivative calculation for parameterized quantum circuits."""
from typing import Literal
from collections.abc import Sequence
from itertools import chain

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector
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
    set_params: dict[ParameterVector, Sequence[int | float]],
    backend: Backend,
    method: Literal['param_shift', 'spsa', 'lin_comb', 'fin_diff'] = 'param_shift'
) -> np.ndarray:
    """
    !TODO: write docs

    Args:
        circuit (QuantumCircuit): _description_
        set_params (dict[ParameterVector, Sequence[int  |  float]]): _description_
        backend (Backend): _description_
        method (Literal[&#39;param_shift&#39;, &#39;spsa&#39;, &#39;lin_comb&#39;, &#39;fin_diff&#39;], optional):
        _description_. Defaults to 'param_shift'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
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

    param_vectors, param_values = zip(*list(set_params.items()))

    params, values = list(chain(*[pv.params for pv in param_vectors])), list(chain(*param_values))

    grads = gradient_calc.run([circuit], [values], [params]).result().gradients[0]
    return _param_grads_to_jacobian(grads, num_possible_values)
