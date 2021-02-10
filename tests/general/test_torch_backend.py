import torch
import pytest
from kymatio.backend.torch_backend import ModulusStable, modulus


def test_modulus(random_state=42):
    """
    Tests the stability and differentiability of modulus
    """

    x = torch.randn(100, 4, 128, dtype=torch.cfloat, requires_grad=True)
    x_grad = x.clone()
    x_abs = modulus(x)

    x_grad = x.real / x_abs + 1j * x.imag / x_abs

    class FakeContext:
        def save_for_backward(self, *args):
            self.saved_tensors = args

    ctx = FakeContext()
    y = ModulusStable.forward(ctx, x)
    y_grad = torch.ones_like(y)
    x_grad_manual = ModulusStable.backward(ctx, y_grad)
    assert torch.allclose(x_grad_manual, x_grad)
