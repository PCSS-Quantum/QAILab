"""Test the backward pass"""
import numpy as np

from qlauncher import QuantumLauncher, Result
from qlauncher.routines.qiskit_routines import QiskitBackend

from qailab.circuit import build_circuit, RotationalEncoder
from qailab.circuit.utils import assign_input_weight
from qailab.qlauncher import CircuitProblem, BackwardPass


def _prepare_circ():
    c = build_circuit(2, [RotationalEncoder('x', 'input'), RotationalEncoder('y', 'weight')])
    return CircuitProblem(c, 'test')


def test_runs_backward():
    """Test if backward pass returns correct results"""
    circp = _prepare_circ()

    layer_input = [2, 1]
    weights = [3, 7]
    algo = BackwardPass()
    be = QiskitBackend('local_simulator')

    ql = QuantumLauncher(circp, algo, be)

    res = ql.run(
        auto_bind=False,
        parameters=assign_input_weight(circp.instance, layer_input, weights)
    )
    assert isinstance(res, Result)
    assert isinstance(res.result, dict)

    for v in res.result.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (2, 4)
