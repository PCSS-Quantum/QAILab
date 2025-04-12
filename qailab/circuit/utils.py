"""Utility functions for circuits."""
from collections.abc import Sequence
from typing import Literal

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def param_map(params: Sequence[Parameter], values: Sequence[float] | np.ndarray) -> dict[Parameter, float]:
    """
    Map values to params.

    Args:
        params (Sequence[Parameter]): Circuit parameters.
        values (Sequence[float]): Values for each parameter.

    Returns:
        dict[Parameter,float]: Param mapping.
    """
    return dict(zip(params, values))


def filter_params(circuit: QuantumCircuit, param_type: Literal['input', 'weight']) -> Sequence[Parameter]:
    """
    Get params of a circuit that match a given type.

    Args:
        circuit (QuantumCircuit): Parameterized circuit.
        param_type (Literal[&#39;input&#39;, &#39;weight&#39;]): Parameter type.

    Returns:
        Sequence[Parameter]: Parameters of type param_type.
    """
    return [p for p in circuit.parameters if p.name.startswith(param_type)]


def assign_input_weight(
    circuit: QuantumCircuit,
    inputs: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray
) -> dict[Parameter, float]:
    """
    Generate assignment of input and weight parameters for a given circuit.

    Args:
        circuit (QuantumCircuit): Parameterized circuit.
        inputs (Sequence[float] | np.ndarray): Input values.
        weights (Sequence[float] | np.ndarray): Weight values.

    Returns:
        dict[Parameter, float]: Combined parameter assignment.
    """
    return {
        **param_map(filter_params(circuit, 'input'), inputs),
        **param_map(filter_params(circuit, 'weight'), weights),
    }
