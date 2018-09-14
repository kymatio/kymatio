import torch
from torch.autograd import Variable
from scattering.scattering1d.fft_wrapper import fft1d_c2c, ifft1d_c2c_normed


def test_correctness_fft(random_state=42, test_cuda=None):
    """
    Tests whether the FFT computed in GPU and CPU are equal,
    and whether the IFFT(FFT) = identity

    Note that for GPU and CPU, we only check an absolute l_infty error
    of 1e-4  , while for the IFFT we check 1e-5
    """
    torch.manual_seed(random_state)
    x = Variable(torch.randn(64, 40, 1024, 2))
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # checking the equality between CPU and GPU (CPU is "safe")
    x_fft_cpu = fft1d_c2c(x)
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        x_fft_gpu = fft1d_c2c(x_gpu)
        x_fft_gpu2 = Variable(x_fft_gpu.data.cpu())
        assert torch.max(torch.abs(x_fft_gpu2.data - x_fft_cpu.data)) < 1e-4
        x_ifft_gpu = ifft1d_c2c_normed(x_fft_gpu)
        x_ifft_gpu2 = Variable(x_ifft_gpu.data.cpu())
        assert torch.max(torch.abs(x_ifft_gpu2.data - x.data)) < 1e-5
    x_ifft_cpu = ifft1d_c2c_normed(x_fft_cpu)
    assert torch.max(torch.abs(x_ifft_cpu.data - x.data)) < 1e-5


def test_correctness_fft_double(random_state=42, test_cuda=None):
    """
    Tests whether the FFT computed in GPU and CPU are equal,
    and whether the IFFT(FFT) = identity

    Note that for GPU and CPU, we only check an absolute l_infty error
    of 1e-4  , while for the IFFT we check 1e-5
    """
    torch.manual_seed(random_state)
    x = Variable(torch.randn(64, 40, 1024, 2).double())
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # checking the equality between CPU and GPU (CPU is "safe")
    x_fft_cpu = fft1d_c2c(x)
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        x_fft_gpu = fft1d_c2c(x_gpu)
        x_fft_gpu2 = Variable(x_fft_gpu.data.cpu())
        assert torch.max(torch.abs(x_fft_gpu2.data - x_fft_cpu.data)) < 1e-8
        x_ifft_gpu = ifft1d_c2c_normed(x_fft_gpu)
        x_ifft_gpu2 = Variable(x_ifft_gpu.data.cpu())
        assert torch.max(torch.abs(x_ifft_gpu2.data - x.data)) < 1e-10
    x_ifft_cpu = ifft1d_c2c_normed(x_fft_cpu)
    assert torch.max(torch.abs(x_ifft_cpu.data - x.data)) < 1e-10


def test_differentiability_fft(random_state=42, test_cuda=None):
    """
    Checks that the FFT produces gradient.
    This does NOT test the validity of the values of the gradients!
    """
    torch.manual_seed(random_state)
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    x = Variable(torch.randn(64, 40, 1024, 2), requires_grad=True)
    x_fft_cpu = fft1d_c2c(x)
    loss_cpu = torch.sum(x_fft_cpu**2)
    loss_cpu.backward()
    assert torch.max(torch.abs(x.grad.data)) > 0.
    if test_cuda:
        x2 = Variable(torch.randn(64, 40, 1024, 2), requires_grad=True)
        x_fft_gpu = fft1d_c2c(x2)
        loss_gpu = torch.sum(x_fft_gpu**2)
        loss_gpu.backward()
        assert torch.max(torch.abs(x2.grad.data)) > 0.
