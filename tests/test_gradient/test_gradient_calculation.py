"""Test gradient calculation"""
import numpy as np

from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.circuit import build_circuit, RotationalEncoder, CXEntangler
from qlearning.circuit.utils import param_map, filter_params
from qlearning.gradient.gradient_calculation import calculate_jacobian


def _prepare_circuit():
    return build_circuit(4, [RotationalEncoder('x', 'weight'), CXEntangler()])


def _test_output_format(method):
    c = _prepare_circuit()
    be = QiskitBackend('local_simulator')
    res = calculate_jacobian(c, param_map(filter_params(c, 'weight'), [-2, -1, 1, 2]), be, method=method)

    assert isinstance(res, np.ndarray)
    assert res.shape == (4, 2**4)


def test_methods():
    """test if all gradient methods work correctly"""
    for m in ['param_shift', 'spsa', 'lin_comb']:
        _test_output_format(m)
