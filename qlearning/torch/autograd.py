"""Autograd extensions for VQCs"""
import numpy as np

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from quantum_launcher import QuantumLauncher

from qlearning.utils import distribution_to_array
from qlearning.circuit.utils import param_map, filter_params

# * Using template code from torch generates weird linter errors,
# * might have to investigate later, ignoring for now as everything seems to work correctly.


class ExpVQCFunction(Function):  # pylint: disable=abstract-method
    """Class implementing forward and backward calculations for ExpQLayer"""
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
        input_params = filter_params(launcher_forward.problem.instance, 'input')
        weight_params = filter_params(launcher_forward.problem.instance, 'weight')

        fn_in_numpy = fn_in.cpu().detach().numpy()
        # Weights are usually pretty small so we rescale them to <-pi,pi>
        weight_numpy = weight.cpu().detach().numpy() * np.pi

        params = {
            **param_map(input_params, fn_in_numpy),
            **param_map(weight_params, weight_numpy)
        }

        res = launcher_forward.run(parameters=params)
        arr = distribution_to_array(res.distribution)

        t = torch.tensor(arr, dtype=fn_in.dtype, requires_grad=True).to(fn_in.device)

        return t

    @staticmethod
    def setup_context(ctx, inputs, output):
        fn_in, weight, launcher_forward, launcher_backward = inputs
        ctx.save_for_backward(fn_in, weight, output)
        ctx.launcher_forward = launcher_forward
        ctx.launcher_backward = launcher_backward

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

        input_params = filter_params(launcher_backward.problem.instance, 'input')
        weight_params = filter_params(launcher_backward.problem.instance, 'weight')

        fn_in_numpy = fn_in.cpu().detach().numpy()
        weight_numpy = weight.cpu().detach().numpy()

        params = {
            **param_map(input_params, fn_in_numpy),
            **param_map(weight_params, weight_numpy)
        }

        res = launcher_backward.run(parameters=params, auto_bind=False)

        out_grad_numpy = grad_output.cpu().detach().numpy()

        grad_input = res.result['input'] @ out_grad_numpy
        grad_weight = res.result['weights'] @ out_grad_numpy

        return (
            torch.tensor(grad_input, dtype=fn_in.dtype).to(fn_in.device),
            torch.tensor(grad_weight, dtype=weight.dtype).to(fn_in.device),
            None, None
        )
