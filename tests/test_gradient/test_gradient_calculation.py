"""Test gradient calculation"""
import numpy as np

from qlauncher.routines.qiskit import QiskitBackend

from qailab.circuit import build_circuit, RotationalEncoder, CXEntangler, AmplitudeEncoder
from qailab.circuit.utils import param_map, filter_params
from qailab.gradient.gradient_calculation import calculate_jacobian


def _prepare_circuit():
    return build_circuit(4, [AmplitudeEncoder('input'), RotationalEncoder('x', 'weight'), CXEntangler()])


def _test_output_format(method):
    c = _prepare_circuit()
    be = QiskitBackend('local_simulator')
    input_map = param_map(filter_params(c, 'input'), [1] * 16)
    weight_map = param_map(filter_params(c, 'weight'), [-2, -1, 1, 2])

    res = calculate_jacobian(
        c.assign_parameters(input_map),
        weight_map, be, method=method)

    assert isinstance(res, np.ndarray)
    assert res.shape == (4, 2**4)

    res = calculate_jacobian(
        c.assign_parameters(weight_map),
        input_map, be, method=method)

    assert isinstance(res, np.ndarray)
    assert res.shape == (16, 2**4)


def test_methods():
    """test if all gradient methods work correctly"""
    for m in ['param_shift', 'spsa', 'lin_comb']:
        _test_output_format(m)
