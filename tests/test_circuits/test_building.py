import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

from qlearning.circuit.circuit_builder import build_circuit

from qlearning.circuit.layer_blocks import CXEntangler, RxWeight, RyWeight, RzWeight
from qlearning.circuit.encoding_blocks import RxEncoder, RyEncoder, RzEncoder


from qlearning.circuit.measurement_blocks import FirstQubitMeasurement, AllQubitMeasurement


def test_builder():
    res = build_circuit(4, [RxWeight, CXEntangler], RyEncoder, FirstQubitMeasurement)
    assert isinstance(res, tuple)
    assert len(res) == 3

    assert isinstance(res[0], QuantumCircuit)
    assert isinstance(res[1], list)
    assert len(res[1]) == 1
    assert isinstance(res[2], ParameterVector)
    for v in res[1]:
        assert isinstance(v, ParameterVector)

    c, w, x = res

    assert c.num_qubits == 4
    assert c.num_parameters == 8  # 4 for weight, 4 for input vector
    assert sum(dict(c.count_ops()).values()) == 4  # check if all blocks get added as gates


def test_first_qubit_measurements():
    c, w, x = build_circuit(3, [RyWeight, CXEntangler], RzEncoder, FirstQubitMeasurement)
    res = Sampler().run(
        c.assign_parameters(
            {
                w[0]: np.random.uniform(-1, 1, (3,)),
                x: [1, -1, 2]
            }
        )
    )

    assert len(res.result().quasi_dists[0]) == 2


def test_all_qubit_measurement():
    c, w, x = build_circuit(3, [RzWeight, CXEntangler], RxEncoder, AllQubitMeasurement)
    res = Sampler().run(
        c.assign_parameters(
            {
                w[0]: np.random.uniform(-1, 1, (3,)),
                x: [1, -1, 2]
            }
        )
    )

    assert len(res.result().quasi_dists[0]) == 2**3
