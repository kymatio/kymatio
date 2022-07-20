import torch
import pytest
from kymatio.backend.torch_backend import ModulusStable, TorchBackend


def test_modulus(random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    backend = TorchBackend
    x = torch.randn(100, 4, 128, 2, requires_grad=True)
    x_grad = x.clone()
    x_abs = backend.modulus(x)[..., 0]

    x_grad[..., 0] = x[..., 0] / x_abs
    x_grad[..., 1] = x[..., 1] / x_abs

    class FakeContext:
        def save_for_backward(self, *args):
            self.saved_tensors = args

    ctx = FakeContext()
    y = ModulusStable.forward(ctx, x)
    y_grad = torch.ones_like(y)
    x_grad_manual = ModulusStable.backward(ctx, y_grad)
    assert torch.allclose(x_grad_manual, x_grad)


def test_reshape():
    backend = TorchBackend

    # 1D
    x = torch.randn(3, 5, 16)
    y = backend.reshape_input(x, signal_shape=(16,))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=1)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16)
    assert torch.allclose(x, xbis)

    # 2D
    x = torch.randn(3, 5, 16, 16)
    y = backend.reshape_input(x, signal_shape=(16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=2)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16, 16)
    assert torch.allclose(x, xbis)

    # 3D
    x = torch.randn(3, 5, 16, 16, 16)
    y = backend.reshape_input(x, signal_shape=(16, 16, 16))
    xbis = backend.reshape_output(y, batch_shape=(3, 5), n_kept_dims=3)
    assert backend.shape(x) == x.shape
    assert y.shape == (15, 1, 16, 16, 16)
    assert torch.allclose(x, xbis)
