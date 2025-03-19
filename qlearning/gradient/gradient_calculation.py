from typing import Literal
from collections.abc import Sequence
from itertools import chain

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


def calculate_weight_gradients(
    circuit,
    weight_params: dict[ParameterVector, Sequence[int | float]],
    backend: Backend,
    method: Literal['param_shift', 'spsa', 'lin_comb', 'fin_diff'] = 'param_shift'
) -> dict[ParameterVector, Sequence[int | float]]:
    gradient_type = {
        'param_shift': ParamShiftSamplerGradient,
        'spsa': SPSASamplerGradient,
        'lin_comb': LinCombSamplerGradient,
        'fin_diff': FiniteDiffSamplerGradient
    }.get(method, None)

    if gradient_type is None:
        raise ValueError(f"Unsupported method {method}")

    if not isinstance(backend, QiskitBackend):
        raise ValueError("Only qiskit backends are supported.")

    """
    ! TODO: change this sampler to backend sampler when adapter is fixed!!!!
    """
    gradient_calc: BaseSamplerGradient = gradient_type(Sampler())

    param_vectors, param_values = zip(*list(weight_params.items()))

    params, values = list(chain(*[pv.params for pv in param_vectors])), list(chain(*param_values))

    grads = gradient_calc.run(circuit, [values], [params]).result().gradients
    print(grads)
    return dict()
