"""Test if circuit builder circuits run with passes"""
from qlauncher import QuantumLauncher, Result
from qlauncher.routines.qiskit_routines import QiskitBackend

from qailab.circuit import RotationalEncoder, build_circuit
from qailab.circuit.utils import assign_input_weight
from qailab.qlauncher import CircuitProblem, ForwardPass


def _prepare_circ():
    c = build_circuit(2, [RotationalEncoder('x', 'input'), RotationalEncoder('y', 'weight')])
    return CircuitProblem(c, 'test')


def test_runs_forward():
    """Test forward pass integration"""
    circp = _prepare_circ()
    c = circp.instance
    algo = ForwardPass()
    be = QiskitBackend('local_simulator')

    ql = QuantumLauncher(circp, algo, be)
    m = assign_input_weight(
        c,
        [0, 0],
        [0, 0]
    )
    res = ql.run(
        parameters=m,
        initial_state=[0, 1, 0, 0]
    )
    assert isinstance(res, Result)
    assert isinstance(res.distribution, dict)
    assert res.distribution[(0, 1)] == 1
