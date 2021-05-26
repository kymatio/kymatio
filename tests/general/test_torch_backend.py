import torch
import pytest


def test_modulus(random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    x = torch.randn(1000, 40, 128, dtype=torch.complex128, requires_grad=True)
    xd = x.detach()
    xv = torch.view_as_real(xd)
    x_abs = torch.abs(xd)

    xv_grad = xv.clone()
    xv_grad[..., 0] = xv[..., 0] / x_abs
    xv_grad[..., 1] = xv[..., 1] / x_abs
    xv_grad = torch.view_as_complex(xv_grad)

    y = torch.abs(x)
    y_grad = torch.ones_like(y)
    y.backward(gradient=y_grad)
    x_grad_manual = x.grad
    assert torch.allclose(x_grad_manual, xv_grad)
