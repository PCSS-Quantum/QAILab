import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

from qlearning.circuit.base import CircuitBlock
from qlearning.circuit.layer_blocks import CXEntangler
from qlearning.circuit.encoding_blocks import RotationalEncoder
from qlearning.circuit.circuit_builder import build_circuit


def test_builder():
    res = build_circuit(4, [RotationalEncoder('y')], [RotationalEncoder('x'), CXEntangler()], [0])
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


class AuxBlock(CircuitBlock):
    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        c = QuantumCircuit(num_qubits + 1)
        for i in range(num_qubits+1):
            c.x(i)
        return c


def test_block_with_aux_qubits():
    """Test if blocks using more qubits than num_qubits get added correctly"""
    c = QuantumCircuit(2)
    b = AuxBlock(name="e")
    b.add_to_circuit(c)

    b2 = AuxBlock(name="e")
    b2.add_to_circuit(c, [0, 1])

    assert c.num_qubits == 4


def test_first_qubit_measurements():
    c, x, w = build_circuit(3, [RotationalEncoder('z')], [RotationalEncoder('y'), CXEntangler()], [0])
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
    c, x, w = build_circuit(3, [RotationalEncoder('x')], [RotationalEncoder('z'), CXEntangler()])
    res = Sampler().run(
        c.assign_parameters(
            {
                w[0]: np.random.uniform(-1, 1, (3,)),
                x[0]: [1, -1, 2]
            }
        )
    )

    assert len(res.result().quasi_dists[0]) == 2**3


def test_param_block_can_be_added_twice():
    """Test if a block added twice generates only one instance of parameter vectors to set later"""
    b = RotationalEncoder('x')
    _, x, _ = build_circuit(3, [b, b])

    assert x == [b.parameter_vector]
