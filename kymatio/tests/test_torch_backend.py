import torch
import pytest
from ..backend.torch_backend import ModulusStable, modulus

def test_modulus(device, random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    if backend.name == 'torch':
        x = torch.randn(100, 4, 128, 2, requires_grad=True, device=device)
        x_abs = modulus(x)
        x_grad = x / x_abs[..., 0].unsqueeze(dim=-1)

        class FakeContext:
            def save_for_backward(self, *args):
                self.saved_tensors = args

        ctx = FakeContext()
        y = ModulusStable.forward(ctx, x)
        y_grad = torch.ones_like(y)
        x_grad_manual = ModulusStable.backward(ctx, y_grad)
        assert torch.allclose(x_grad_manual, x_grad)