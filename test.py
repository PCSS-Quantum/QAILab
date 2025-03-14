import torch
import torch.nn as nn


class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass.
        ctx is a context object to store information for backward.
        input is a tensor.
        """
        # Save input for backward
        ctx.save_for_backward(input)

        # Example forward op: multiply by 2
        output = input * 2
        print("Forward pass:")
        print("  Input:", input)
        print("  Output:", output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.
        grad_output is the gradient of the loss w.r.t. the output of forward().
        Must return as many tensors as there were inputs to forward that require grad.
        """
        # Retrieve the saved input
        (input_saved,) = ctx.saved_tensors

        print("\nBackward pass:")
        print("  grad_output (dL/dOut):", grad_output)
        print("  Saved input (for reference):", input_saved)

        # Derivative of output = (input * 2) w.r.t input is 2,
        # so grad_input = grad_output * 2
        grad_input = grad_output * 2
        print("  grad_input (dL/dIn):", grad_input)

        # Return gradient for each input to forward()
        return grad_input


class MyCustomLayer(nn.Module):
    def __init__(self):
        super(MyCustomLayer, self).__init__()
        # No learnable parameters in this trivial example

    def forward(self, x):
        # Apply our custom autograd function
        return MyCustomFunction.apply(x)


def main():
    # Instantiate custom layer
    layer = MyCustomLayer()

    # Create some random input, requires_grad=True to track gradients
    x = torch.randn(3, requires_grad=True)
    print("Initial Input x:", x)

    # Forward pass
    y = layer(x)
    print("\nResult of custom layer, y:", y)

    # For demonstration, let's make a scalar loss
    loss = y.sum()
    print("\nScalar loss (sum of y):", loss.item())

    # Backward pass (this triggers MyCustomFunction.backward)
    loss.backward()

    # After backward, x.grad holds dLoss/dx
    print("\nGradient wrt x (dLoss/dx):", x.grad)


if __name__ == "__main__":
    main()
