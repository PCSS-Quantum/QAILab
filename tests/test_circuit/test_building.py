import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

from qlearning.circuit.circuit_builder import build_circuit

from qlearning.circuit.layer_blocks import CXEntangler
from qlearning.circuit.encoding_blocks import RotationalEncoder


from qlearning.circuit.measurement_blocks import FirstQubitMeasurement, AllQubitMeasurement


def test_builder():
    res = build_circuit(4, [RotationalEncoder('y')], [RotationalEncoder('x'), CXEntangler()],  FirstQubitMeasurement())
    assert isinstance(res, tuple)
    assert len(res) == 3

    assert isinstance(res[0], QuantumCircuit)
    assert isinstance(res[1], list)
    assert len(res[1]) == 1
    assert isinstance(res[2], list)
    for v in res[1] + res[2]:
        assert isinstance(v, ParameterVector)

    c, _, _ = res

    assert c.num_qubits == 4
    assert c.num_parameters == 8  # 4 for weight, 4 for input vector
    assert sum(dict(c.count_ops()).values()) == 4  # check if all blocks get added as gates


def test_first_qubit_measurements():
    c, x, w = build_circuit(3, [RotationalEncoder('z')], [RotationalEncoder('y'), CXEntangler()], FirstQubitMeasurement())
    res = Sampler().run(
        c.assign_parameters(
            {
                w[0]: np.random.uniform(-1, 1, (3,)),
                x[0]: [1, -1, 2]
            }
        )
    )

    assert len(res.result().quasi_dists[0]) == 2


def test_all_qubit_measurement():
    c, x, w = build_circuit(3, [RotationalEncoder('x')], [RotationalEncoder('z'), CXEntangler()], AllQubitMeasurement())
    res = Sampler().run(
        c.assign_parameters(
            {
                w[0]: np.random.uniform(-1, 1, (3,)),
                x[0]: [1, -1, 2]
            }
        )
    )

    assert len(res.result().quasi_dists[0]) == 2**3
