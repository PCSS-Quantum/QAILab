"""Test backward pass integration with torch autograd"""
import torch

import numpy as np

from qlauncher import QuantumLauncher
from qlauncher.routines.qiskit_routines import QiskitBackend

from qailab.qlauncher import CircuitProblem, ForwardPass, BackwardPass

from qailab.torch.autograd import ExpVQCFunction, ArgMax
from qailab.circuit import RotationalEncoder, build_circuit
from qailab.circuit.utils import filter_params


def _prep_circuit():
    return build_circuit(
        2,
        [
            RotationalEncoder('x', 'input'),
            RotationalEncoder('x', 'weight')
        ])


def _run_forward():

    circ = _prep_circuit()
    circ_p = CircuitProblem(circ, 'test')

    input_t = torch.tensor(np.zeros((len(filter_params(circ, 'input')),)), dtype=torch.float32, requires_grad=False)
    weights = torch.tensor([0.2] + [0.0001] * (len(filter_params(circ, 'weight')) - 1), dtype=torch.float32, requires_grad=True)

    backend = QiskitBackend('local_simulator')

    launcher_forward = QuantumLauncher(circ_p, ForwardPass(shots=1024), backend)
    launcher_backward = QuantumLauncher(circ_p, BackwardPass('param_shift', shots=1024), backend)

    apply = ExpVQCFunction.apply

    def res(i, w): return apply(i, w, launcher_forward, launcher_backward)
    return res, input_t, weights


def test_autograd_forward():
    """Test if autograd.Function implementation of forward works correctly"""
    res, i, w = _run_forward()

    out = res(i, w)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4,)


def test_autograd_backward():
    """Test if gradients are calculated on backward pass"""
    res, i, w = _run_forward()
    out = res(i, w)
    loss_fn = torch.nn.MSELoss()

    exp = torch.Tensor([1, 0, 0, 0])

    l = loss_fn(out, exp)

    l.backward()

    # Check if weights receive gradient
    assert w.grad is not None


def test_autograd_opt():
    res, i, w = _run_forward()

    opt = torch.optim.Adam([w], lr=0.05)
    loss_fn = torch.nn.MSELoss()

    exp = torch.Tensor([0, 1, 0, 0])

    loss_items = []

    for _ in range(100):
        opt.zero_grad()
        out = res(i, w)
        l = loss_fn(out, exp)
        loss_items.append(l.item())

        l.backward()
        opt.step()

    assert loss_items[-1] < 0.01


def test_autograd_diff_argmax():
    t = torch.tensor([0.0, 1.0, 0.5])
    argmax = ArgMax.apply(t)

    assert isinstance(argmax, torch.Tensor)

    assert argmax == 1
