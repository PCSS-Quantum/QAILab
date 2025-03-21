"""Test the backward pass"""
import numpy as np

from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.circuit import build_circuit, RotationalEncoder
from qlearning.circuit.utils import param_map, filter_params
from qlearning.qlauncher import CircuitProblem, BackwardPass


def _prepare_circ():
    c = build_circuit(2, [RotationalEncoder('x', 'input'), RotationalEncoder('y', 'weight')])
    return CircuitProblem(c, 'test')


def test_runs_backward():
    """Test if backward pass returns correct results"""
    circp = _prepare_circ()
    input_params = filter_params(circp.instance, 'input')
    weight_params = filter_params(circp.instance, 'weight')

    layer_input = [2, 1]
    weights = [3, 7]
    algo = BackwardPass()
    be = QiskitBackend('local_simulator')

    ql = QuantumLauncher(circp, algo, be)

    res = ql.run(
        auto_bind=False,
        parameters={
            **param_map(input_params, layer_input),
            **param_map(weight_params, weights)
        }
    )
    assert isinstance(res, Result)
    assert isinstance(res.result, dict)

    for v in res.result.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (2, 4)
