from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.circuit import RotationalEncoder, build_circuit
from qlearning.qlauncher import CircuitProblem, ForwardPass


def prepare_circ():
    c, x, w = build_circuit(2, [RotationalEncoder('x')], [RotationalEncoder('y')])
    return CircuitProblem(c, 'test'), x, w


def test_runs_forward():
    circp, x, w = prepare_circ()
    algo = ForwardPass()
    be = QiskitBackend('local_simulator')

    ql = QuantumLauncher(circp, algo, be)

    res = ql.run(
        parameters={
            x[0]: [0, 0],
            w[0]: [0, 0]
        },
        initial_state=[0, 1, 0, 0]
    )
    assert isinstance(res, Result)
    assert isinstance(res.distribution, dict)
    assert res.distribution[(0, 1)] == 1
