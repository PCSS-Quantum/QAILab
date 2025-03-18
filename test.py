from qlearning.circuit.circuit_builder import build_circuit

from qlearning.circuit.layer_blocks import CXEntangler, RyWeight
from qlearning.circuit.encoding_blocks import RxEncoder

from qlearning.circuit.measurement_blocks import FirstQubitMeasurement


c, w, x = build_circuit(
    4,
    [
        RyWeight,
        CXEntangler,
        RyWeight,
        CXEntangler
    ],
    RxEncoder,
    FirstQubitMeasurement
)

print(c.assign_parameters({wv: [0]*4 for wv in w} | {x: [1]*4}).decompose().draw())

print(dict(c.count_ops()))
print(CXEntangler(4)._circuit.draw())
