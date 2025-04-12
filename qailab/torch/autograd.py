"""Autograd functions for VQCs"""
import numpy as np

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from quantum_launcher import QuantumLauncher

from qailab.utils import distribution_to_array
from qailab.circuit.utils import filter_params, assign_input_weight

# * Using template code from torch generates weird linter errors,
# * might have to investigate later, ignoring for now as everything seems to work correctly.


def _is_batch_input(t: torch.Tensor, num_input_params: int) -> bool:
    return len(t.shape) == 2 and t.shape[1] == num_input_params


class ExpVQCFunction(Function):  # pylint: disable=abstract-method
    """Class implementing forward and backward calculations for ExpQLayer"""
    @staticmethod
    def _forward_single(
        fn_in: torch.Tensor,
        weight: torch.Tensor,
        launcher_forward: QuantumLauncher,
    ) -> torch.Tensor:
        fn_in_numpy = fn_in.cpu().detach().numpy()
        weight_numpy = weight.cpu().detach().numpy()

        params = assign_input_weight(
            launcher_forward.problem.instance,
            fn_in_numpy,
            weight_numpy
        )
        res = launcher_forward.run(parameters=params)
        arr = distribution_to_array(res.distribution)
        t = torch.tensor(arr, dtype=fn_in.dtype, requires_grad=True).to(fn_in.device)

        return t

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        fn_in: torch.Tensor,
        weight: torch.Tensor,
        launcher_forward: QuantumLauncher,
        launcher_backward: QuantumLauncher  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Calculation of forward pass.

        Args:
            fn_in (torch.Tensor): Input tensor.
            weight (torch.Tensor): Layer weights.
            launcher_forward (QuantumLauncher): Qlauncher with forward pass algorithm.
            launcher_backward (QuantumLauncher):
            Qlauncher with backward pass algorithm.
            Not used in forward, but needed here as it will get passed to setup_context()

        Returns:
            torch.Tensor: Distribution of forward pass.
        """

        is_batch = _is_batch_input(fn_in, len(filter_params(launcher_forward.problem.instance, 'input')))

        if is_batch:
            return torch.stack([ExpVQCFunction._forward_single(single_in, weight, launcher_forward) for single_in in fn_in])
        return ExpVQCFunction._forward_single(fn_in, weight, launcher_forward)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Called after forward, saves args from forward to be later used in backward.

        Args:
            ctx: Context object that holds information.
            inputs: args to forward()
            outputs: outputs from forward()
        """
        fn_in, weight, launcher_forward, launcher_backward = inputs
        ctx.save_for_backward(fn_in, weight, output)
        ctx.launcher_forward = launcher_forward
        ctx.launcher_backward = launcher_backward
        ctx.is_batch = _is_batch_input(fn_in, len(filter_params(launcher_forward.problem.instance, 'input')))

    @staticmethod
    def _backward_single(
        fn_in,
        weight,
        launcher_backward,
        grad_output
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fn_in_numpy = fn_in.cpu().detach().numpy()
        weight_numpy = weight.cpu().detach().numpy()

        params = assign_input_weight(
            launcher_backward.problem.instance,
            fn_in_numpy,
            weight_numpy
        )

        res = launcher_backward.run(parameters=params, auto_bind=False)

        out_grad_numpy = grad_output.cpu().detach().numpy()

        grad_input = res.result['input'] @ out_grad_numpy
        # Allow for weightless QNN layers
        grad_weight = res.result['weight'] @ out_grad_numpy if len(res.result['weight']) > 0 else np.array([])

        # Scale gradient values because we are optimizing weights initialized in range <0,2pi>
        return (
            torch.tensor(grad_input, dtype=fn_in.dtype).to(fn_in.device),
            torch.tensor(grad_weight, dtype=weight.dtype).to(fn_in.device) * np.pi,
        )

    @staticmethod
    @once_differentiable
    def backward(  # pylint: disable=arguments-differ
        ctx,
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Calculation of backward pass.

        Args:
            ctx: Context object supplied by autograd. Contains saved tensors and qlaunchers.
            grad_output (torch.Tensor): Grad from next layer.

        Returns:
            tuple[torch.Tensor,torch.Tensor,None,None]:
            Grad for inputs, Grad for weights, rest irrelevant.
            (each forward argument needs to get something, but launchers don't need grad)
        """
        forward_tensors = ctx.saved_tensors
        fn_in, weight = forward_tensors[:2]
        launcher_backward = ctx.launcher_backward

        if not ctx.is_batch:
            return *ExpVQCFunction._backward_single(fn_in, weight, launcher_backward, grad_output), None, None

        input_grads, weight_grads = [], []
        for in_single, grad_single in zip(fn_in, grad_output):
            igrad, wgrad = ExpVQCFunction._backward_single(in_single, weight, launcher_backward, grad_single)
            input_grads.append(igrad)
            weight_grads.append(wgrad)

        return torch.stack(input_grads), torch.stack(weight_grads), None, None


class ArgMax(Function):  # pylint: disable=abstract-method
    """
    ArgMax function. Propagates the sum of gradient on argmax index, rest is zero.

    https://discuss.pytorch.org/t/differentiable-argmax/33020
    """
    @staticmethod
    def forward(fn_in):  # pylint: disable=arguments-differ
        """
        Forward run.

        Args:
            fn_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: First index(es) of maximum elements.
        """
        return torch.tensor(torch.argmax(fn_in, dim=-1, keepdim=True), dtype=fn_in.dtype, requires_grad=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors for backward pass"""
        ctx.save_for_backward(*inputs, output)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
        ctx,
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """
        Calculation of backward pass.

        Args:
            ctx: Context object supplied by autograd.
            grad_output (torch.Tensor): Grad from next layer.

        Returns:
            tuple[torch.Tensor]: Grad w.r.t. input.
        """
        fn_in, idx = ctx.saved_tensors
        grad_input = torch.zeros(fn_in.shape, device=fn_in.device, dtype=fn_in.dtype)
        grad_input.scatter_(-1, torch.tensor(idx, dtype=torch.int64), grad_output.sum(-1, keepdim=True))
        return (grad_input,)
