"""Test the circuit builder"""
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

from qlearning.circuit.base import CircuitBlock
from qlearning.circuit.layer_blocks import CXEntangler
from qlearning.circuit.encoding_blocks import RotationalEncoder
from qlearning.circuit.circuit_builder import build_circuit
from qlearning.circuit.utils import param_map, filter_params


def test_builder():
    """Basic test"""
    res = build_circuit(4, [RotationalEncoder('y', 'input'), RotationalEncoder('x', 'weight'), CXEntangler()], [0])
    assert isinstance(res, QuantumCircuit)

    assert res.num_qubits == 4
    assert res.num_parameters == 8  # 4 for weight, 4 for input vector
    assert sum(dict(res.count_ops()).values()) == 4  # check if all blocks get added as gates


class AuxBlock(CircuitBlock):
    """Dummy block using one aux qubit"""

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        c = QuantumCircuit(num_qubits + 1)
        for i in range(num_qubits + 1):
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
    """Test if correct num of qubits get measured"""
    c = build_circuit(3, [RotationalEncoder('z', 'input'), RotationalEncoder('y', 'weight'), CXEntangler()], [0])
    m = {
        **param_map(filter_params(c, 'weight'), np.random.uniform(-1, 1, (3,)).tolist()),
        **param_map(filter_params(c, 'input'), [1, -1, 2])
    }
    res = Sampler().run(
        c.assign_parameters(m)
    )

    assert len(res.result().quasi_dists[0]) == 2


def test_all_qubit_measurement():
    """Test if correct num of qubits get measured"""
    c = build_circuit(3, [RotationalEncoder('x', 'input'), RotationalEncoder('z', 'weight'), CXEntangler()])
    m = {
        **param_map(filter_params(c, 'weight'), np.random.uniform(-1, 1, (3,)).tolist()),
        **param_map(filter_params(c, 'input'), [1, -1, 2])
    }
    res = Sampler().run(
        c.assign_parameters(m)
    )

    assert len(res.result().quasi_dists[0]) == 2**3


def test_param_block_can_be_added_twice():
    """Test if a block added twice generates only one instance of parameter vectors to set later"""
    b = RotationalEncoder('x', 'input')
    c = build_circuit(3, [b, b])

    assert len(filter_params(c, 'input')) == 3
