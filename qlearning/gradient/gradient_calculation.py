"""Gradient and partial derivative calculation methods for parameterized quantum circuits."""
from typing import Literal

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.gradients import (
    BaseSamplerGradient,
    LinCombSamplerGradient,
    SPSASamplerGradient,
    ParamShiftSamplerGradient,
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
    method: Literal['param_shift', 'spsa', 'lin_comb'] = 'param_shift',
    shots: int = 1024
) -> np.ndarray:
    """
    For each parameter calculate partial derivatives w.r.t to each output value (possible measurement).

    Args:
        circuit (QuantumCircuit): Circuit to sample.
        set_params (dict[Parameter, int  |  float]): Parameters for which to calculate derivatives and their current values.
        backend (Backend): Backend to use.
        method (Literal[&#39;param_shift&#39;, &#39;spsa&#39;, &#39;lin_comb&#39;], optional):
        Gradient algorithm to use. Defaults to 'param_shift'.
        shots (int): How many shots to use for Sampler.

    Raises:
        ValueError: For unsupported method or backend.

    Returns:
        np.ndarray: `len(set_params) x 2^measured_qubits` matrix of partial derivatives.
    """
    if not isinstance(backend, QiskitBackend):
        raise ValueError("Only qiskit backends are supported.")

    num_possible_values = 2**circuit.num_clbits

    set_params_amp = {k: v for k, v in set_params.items() if 'Amp_Encoder' in k.name}
    set_params_compatible = {k: v for k, v in set_params.items() if 'Amp_Encoder' not in k.name}

    grads_amp = calculate_gradients_incompatible(circuit.assign_parameters(set_params_compatible), set_params_amp, backend, shots, 0.01)
    grads_compat = calculate_gradients_compatible(circuit.assign_parameters(set_params_amp), set_params_compatible, backend, method, shots)

    grads_amp = dict(zip(set_params_amp.keys(), grads_amp))
    grads_compat = dict(zip(set_params_compatible.keys(), grads_compat))

    grads = grads_amp | grads_compat

    return _param_grads_to_jacobian([grads[k] for k in set_params.keys()], num_possible_values)


def calculate_gradients_compatible(
    circuit: QuantumCircuit,
    set_params: dict[Parameter, int | float],
    backend: QiskitBackend,
    method: Literal['param_shift', 'spsa', 'lin_comb'] = 'param_shift',
    shots: int = 1024
):
    """
    Calculate gradients for parameters in compatible gates.

    Args:
        circuit (QuantumCircuit): Circuit to sample.
        set_params (dict[Parameter, int  |  float]): Parameters for which to calculate derivatives and their current values.
        backend (QiskitBackend): Backend to use.
        method (Literal[&#39;param_shift&#39;, &#39;spsa&#39;, &#39;lin_comb&#39;], optional):
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
    }.get(method, None)

    if gradient_type is None:
        raise ValueError(f"Unsupported method {method}")

    gradient_calc: BaseSamplerGradient = gradient_type(backend.samplerV1)
    if len(set_params) == 0:
        return []
    params, values = zip(*list(set_params.items()))

    grads = gradient_calc.run([circuit], [values], [params], shots=shots).result().gradients[0]
    return grads


def calculate_gradients_incompatible(
    circuit: QuantumCircuit,
    set_params: dict[Parameter, int | float],
    backend: QiskitBackend,
    shots: int = 1024,
    epsilon: float = 0.01
):
    """
    Calculate gradients for parameters in incompatible gates, e.g. amplitude encoding parameters.
    Calculation is done using the standard derivative formula (f(x+h) - f(x))/h

    Args:
        circuit (QuantumCircuit): Circuit to sample.
        set_params (dict[Parameter, int  |  float]): Parameters for which to calculate derivatives and their current values.
        backend (QiskitBackend): Backend to use.
        shots (int): How many shots to use for Sampler.
        epsilon (float): How much to shift the parameter.

    Raises:
        ValueError: For unsupported method or backend.

    Returns:
        np.ndarray: `len(set_params) x 2^measured_qubits` matrix of partial derivatives.
    """
    initial_distribution = backend.samplerV1.run(circuit.assign_parameters(set_params), shots=shots).result().quasi_dists[0]
    shifted_results = []
    for param in set_params:
        distribution = backend.samplerV1.run(
            circuit.assign_parameters(
                set_params | {param: set_params[param] + epsilon}
            ),
            shots=shots).result().quasi_dists[0]

        for k in initial_distribution:
            distribution[k] = distribution.get(k, 0) - initial_distribution[k]

        for k in distribution:
            distribution[k] /= epsilon

        shifted_results.append(distribution)
    return shifted_results
