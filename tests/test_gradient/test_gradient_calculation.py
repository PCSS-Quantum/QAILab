
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.gradient.gradient_calculation import calculate_weight_gradients
from qlearning.circuit import build_circuit, RotationalEncoder, CXEntangler


def prepare_circuit():
    return build_circuit(4, [], [RotationalEncoder('x'), CXEntangler()])


def test_output_format():
    c, _, w = prepare_circuit()
    be = QiskitBackend('local_simulator')
    res = calculate_weight_gradients(c, {w[0]: [0]*4}, be)

    assert isinstance(res, dict)
