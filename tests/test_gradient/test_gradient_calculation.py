"""Test gradient calculation"""
from collections.abc import Sequence

from qiskit.circuit import ParameterVector
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.gradient.gradient_calculation import calculate_partial_derivatives
from qlearning.circuit import build_circuit, RotationalEncoder, CXEntangler


def _prepare_circuit():
    return build_circuit(4, [], [RotationalEncoder('x'), CXEntangler()])


def _test_output_format(method):
    c, _, w = _prepare_circuit()
    be = QiskitBackend('local_simulator')
    res = calculate_partial_derivatives(c, {w[0]: [-2, -1, 1, 2]}, be, method=method)

    assert isinstance(res, dict)
    for k in res:
        assert isinstance(k, ParameterVector)
    for v in res.values():
        assert isinstance(v, Sequence)
        for vv in v:
            assert isinstance(vv, Sequence)


def test_methods():
    """test if all gradient methods work correctly"""
    for m in ['param_shift', 'spsa', 'lin_comb', 'fin_diff']:
        _test_output_format(m)
