""" File for testing utility functions """
from qlearning.utils import number_to_bit_tuple


def test_number_to_bit_tuple():
    """ Basic test """
    assert number_to_bit_tuple(2, 2) == (1, 0)
    assert number_to_bit_tuple(2, 3) == (0, 1, 0)
