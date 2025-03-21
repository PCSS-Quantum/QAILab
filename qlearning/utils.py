""" Utility functions """
import itertools
import numpy as np


def number_to_bit_tuple(number: int, size: int) -> tuple[int, ...]:
    """Changes number to tuple of bits.

    Args:
        number (int): number

    Returns:
        tuple[int]: tuple with bits
    """
    bits = list(map(int, bin(int(number))[2:]))
    bits = [0] * (size - len(bits)) + bits
    return tuple(map(int, bits))


def distribution_to_array(distribution: dict[tuple[int, ...], float]) -> np.ndarray:
    """Changes the distribution into the one dimensional array with size 2^n of respective values for each permutation.

    Args:
        distribution (dict[tuple[int, ...], float]): the distribution in given format.

    Returns:
        np.ndarray: one dimensional array of given distribution.
    """
    size = len(next(iter(distribution.keys())))
    array = np.zeros((2**size,))
    for ind, combination in enumerate(itertools.product((0, 1), repeat=size)):
        array[ind] = distribution[combination]
    return array
