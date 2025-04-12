""" File for testing utility functions """
from qiskit.circuit import QuantumCircuit, Parameter

from qailab.utils import number_to_bit_tuple
from qailab.circuit.utils import param_map, filter_params, assign_input_weight


def test_number_to_bit_tuple():
    """ Basic test """
    assert number_to_bit_tuple(2, 2) == (1, 0)
    assert number_to_bit_tuple(2, 3) == (0, 1, 0)


def test_filter_params():
    """Test if params are filtered correctly"""

    p1, p2, p3 = Parameter("input_p"), Parameter("weight_p"), Parameter("other_p")
    c = QuantumCircuit(1)
    c.rx(p1, 0)
    c.rx(p2, 0)
    c.rx(p3, 0)

    assert filter_params(c, 'input')[0] == p1
    assert filter_params(c, 'weight')[0] == p2


def test_param_map():
    """Test if param map generates correct output"""
    p1, p2, p3 = Parameter("input_p"), Parameter("weight_p"), Parameter("other_p")
    vals = [1, 2, 3]

    assert param_map([p1, p2, p3], vals) == {p1: 1, p2: 2, p3: 3}


def test_assign_input_weight():
    """Test if input and weight are assigned correctly"""
    p1, p2 = Parameter("input_p"), Parameter("weight_p")
    c = QuantumCircuit(1)
    c.rx(p1, 0)
    c.rx(p2, 0)

    assert assign_input_weight(c, [1], [1]) == {p1: 1, p2: 1}
