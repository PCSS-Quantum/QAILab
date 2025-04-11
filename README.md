# <div style='display:flex; align-items:center;gap:0.5rem;'><div><img src='docs/_static/logo.svg'></div>QLearning</div>

## About project

QLearning is a high level library, which enables the user to integrate quantum neural networks of different configurations into their existing PyTorch neural networks.  
The library aims to support a large amount of backends from different quantum computer manufacturers, such as IBM, AQT and Orca Computing, via its integration with Quantum Launcher.

## Installation

You can install QLearning via pip:
```sh
pip install qlearning
```

## Examples and tutorials

### Creating a QNN circut

```python
from qlearning.circuit import build_circuit, RotationalEncoder, CXEntangler, RealAmplitudesBlock

input_encoder = RotationalEncoder('x','input')
quantum_circuit = build_circuit(
    3,
    [
        input_encoder,
        CXEntangler(),
        RealAmplitudesBlock('weight'),
        input_encoder, #You can put the same block twice in different parts of the circuit. It will encode the same parameters
        CXEntangler(),
        RealAmplitudesBlock('weight')
    ],
    measure_qubits=[0,1]
    )
```

### Making a hybrid neural network

```python
import torch.nn as nn
from qlearning.torch import QLayer

qlayer = QLayer(
    quantum_circuit,
    shots = 2048
    )

sequential_net = nn.Sequential(
    nn.Linear(4,qlayer.in_features),
    nn.ReLU()
    qlayer,
    nn.Linear(qlayer.out_features,4),
    nn.Softmax()
)
```

### Training a QModel instance

```python
model = QModel(
    module=sequential_net,
    loss=nn.CrossEntropyLoss(),
    optimizer_type='adam',
    learning_rate=0.001,
    batch_size=4,
    validation_fraction=0.1,
    epochs=10
    )

model.fit(X,y)
```

More advanced tutorials are available in the [QLearning documentation](https://qlearning-84a1ee.gitlab-pages.pcss.pl/index.html)