"""Different implementations of input vector encoding blocks for variational circuits."""
from qailab.circuit.encoding_blocks.rotational import RotationalEncoder, RealAmplitudesBlock
from qailab.circuit.encoding_blocks.amplitude import AmplitudeEncoder

__all__ = ['RotationalEncoder', 'AmplitudeEncoder', 'RealAmplitudesBlock']
