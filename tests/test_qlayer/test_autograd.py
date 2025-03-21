"""Test backward pass integration with torch autograd"""
import torch

import numpy as np

from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.qiskit_routines import QiskitBackend

from qlearning.qlauncher import CircuitProblem, ForwardPass, BackwardPass

from qlearning.torch.autograd import ExpVQCFunction
from qlearning.circuit import RotationalEncoder, build_circuit
from qlearning.circuit.utils import filter_params


def _prep_circuit():
    return build_circuit(
        2,
        [
            RotationalEncoder('x', 'input'),
            RotationalEncoder('z', 'weight'),
            RotationalEncoder('x', 'input'),
            RotationalEncoder('z', 'weight')
        ])


def _run_forward():

    circ = _prep_circuit()
    circ_p = CircuitProblem(circ, 'test')

    input_t = torch.tensor(np.random.uniform(-2, 2, (len(filter_params(circ, 'input')),)), dtype=torch.float32, requires_grad=True)
    weights = torch.tensor(np.random.uniform(-2, 2, (len(filter_params(circ, 'weight')),)), dtype=torch.float32, requires_grad=True)

    backend = QiskitBackend('local_simulator')

    launcher_forward = QuantumLauncher(circ_p, ForwardPass(shots=1024), backend)
    launcher_backward = QuantumLauncher(circ_p, BackwardPass('spsa', shots=1024), backend)

    apply = ExpVQCFunction.apply

    res = apply(input_t, weights, launcher_forward, launcher_backward)
    return res, input_t, weights


def test_autograd_forward():
    """Test if autograd.Function implementation of forward works correctly"""
    res, _, _ = _run_forward()

    assert isinstance(res, torch.Tensor)
    assert res.shape == (4,)


def test_autograd_backward():
    """Test if gradients are calculated on backward pass"""
    res, i, w = _run_forward()

    loss_fn = torch.nn.MSELoss()

    exp = torch.Tensor([1, 0, 0, 0])

    l = loss_fn(res, exp)

    l.backward()

    # Check if input and weights receive gradient
    assert i.grad is not None
    assert w.grad is not None
