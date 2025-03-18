""" Utility functions """


def number_to_bit_tuple(number: int, size: int) -> tuple[int, ...]:
    """Changes number to tuple of bits.

    Args:
        number (int): number

    Returns:
        tuple[int]: tuple with bits
    """
    bits = list(map(int, bin(number)[2:]))
    bits = [0] * (size - len(bits)) + bits
    return tuple(map(int, bits))
