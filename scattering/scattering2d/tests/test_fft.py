import torch
from torch.autograd import Variable
import numpy as np
from scattering.scattering2d.FFT import fft_c2c, ifft_c2c, ifft_c2r, fft_r2c


def fft_c2c_cpu(x, inverse=False):
    """
    Computes the FFT in 2D of a complex tensor, along the last two axis.
    Help function for the tests.
    """
    # take it to numpy
    x_np = x.numpy()
    # convert to complex
    if x_np.dtype == 'float32':
        original_type = 'float32'
        x_np.dtype = 'complex64'
    elif x_np.dtype == 'float64':
        original_type = 'float64'
        x_np.dtype = 'complex128'
    else:
        raise ValueError('fft_c2c_cpu supports only floats 32 and 64, but got',
                         x_np.dtype)
    # remove the last axis
    x_np = x_np.reshape(x_np.shape[:-1])
    # perform the FFT operation
    if inverse:  # unnormalized
        res = np.fft.ifft2(x_np) * float(x_np.shape[-1] * x_np.shape[-2])
    else:
        res = np.fft.fft2(x_np)
    # Make sure that the types of x_np and res match
    res = np.asarray(res, dtype=x_np.dtype)
    # Separate the real and imaginary parts of res
    # res = res[..., np.newaxis]
    # res.dtype = original_type
    res2 = np.zeros(res.shape + (2,), dtype=original_type)
    res2[..., 0] = np.real(res)
    res2[..., 1] = np.imag(res)
    # move it to torch
    output = torch.from_numpy(res2)
    return output


def test_correctness_fft_c2c_float(random_state=42, test_cuda=None):
    """
    Tests whether the FFT computed in GPU and CPU are equal,
    and whether the IFFT(FFT) = identity

    Note that for GPU and CPU, we only check an absolute l_infty error
    of 1e-3, while for the IFFT we check 1e-5
    """
    torch.manual_seed(random_state)
    x = Variable(torch.randn(32, 40, 64, 64, 2))
    factor = x.shape[-2] * x.shape[-3]
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # checking the equality between CPU and GPU (CPU is "safe")
    x_fft_cpu = Variable(fft_c2c_cpu(x.data))
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        x_fft_gpu = fft_c2c(x_gpu)
        x_fft_gpu2 = Variable(x_fft_gpu.data.cpu())
        assert torch.max(torch.abs(x_fft_gpu2.data - x_fft_cpu.data)) < 1e-3
        x_ifft_gpu = ifft_c2c(x_fft_gpu)
        x_ifft_gpu2 = Variable(x_ifft_gpu.data.cpu())
        assert torch.max(torch.abs(x_ifft_gpu2.data / factor - x.data)) < 1e-5
    x_ifft_cpu = fft_c2c_cpu(x_fft_cpu.data, inverse=True)
    assert torch.max(torch.abs(x_ifft_cpu / factor - x.data)) < 1e-5


def test_correctness_fft_c2c_double(random_state=42, test_cuda=None):
    """
    Tests whether the FFT computed in GPU and CPU are equal,
    and whether the IFFT(FFT) = identity

    Note that for GPU and CPU, we only check an absolute l_infty error
    of 1e-10, while for the IFFT we check 1e-10
    """
    torch.manual_seed(random_state)
    x = Variable(torch.randn(16, 20, 64, 64, 2).double())
    factor = x.shape[-2] * x.shape[-3]
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # checking the equality between CPU and GPU (CPU is "safe")
    x_fft_cpu = fft_c2c_cpu(x.data)
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        x_fft_gpu = fft_c2c(x_gpu)
        x_fft_gpu2 = Variable(x_fft_gpu.data.cpu())
        assert torch.max(torch.abs(x_fft_gpu2.data - x_fft_cpu)) < 1e-10
        x_ifft_gpu = ifft_c2c(x_fft_gpu)
        x_ifft_gpu2 = Variable(x_ifft_gpu.data.cpu())
        assert torch.max(torch.abs(x_ifft_gpu2.data / factor - x.data)) < 1e-10
    x_ifft_cpu = fft_c2c_cpu(x_fft_cpu, inverse=True)
    assert torch.max(torch.abs(x_ifft_cpu / factor - x.data)) < 1e-10


def test_correctness_fft_c2r_float(random_state=42, test_cuda=None):
    """
    Tests whether the accelerated C2R ifft is correct
    """
    torch.manual_seed(random_state)
    # start with a real vector
    x_real = torch.randn(32, 40, 64, 64, 2)
    x_real[..., 1] = 0.
    # compute its FFT, which ensures hermitian symmetry
    x = Variable(fft_c2c_cpu(x_real))
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # compute its IFFT in full
    x_back_full_cpu = fft_c2c_cpu(x.data, inverse=True)
    assert torch.max(torch.abs(x_back_full_cpu)[..., 1]) < 1e-7
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        # full IFFT
        x_back_full_gpu = ifft_c2c(x_gpu)
        x_back_full_gpu2 = x_back_full_gpu.data.cpu()
        assert torch.max(torch.abs(x_back_full_gpu2 - x_back_full_cpu)) < 1e-2
        assert torch.max(torch.abs(x_back_full_gpu2)[..., 1]) < 1e-2
        # quick IFFT
        x_back_quick_gpu = ifft_c2r(x_gpu)
        x_back_quick_gpu2 = x_back_quick_gpu.data.cpu()
        assert torch.max(torch.abs(x_back_quick_gpu2 - x_back_full_gpu2[..., 0])) < 1e-5


def test_correctness_fft_c2r_double(random_state=42, test_cuda=None):
    """
    Tests whether the accelerated C2R ifft is correct for double precision
    """
    torch.manual_seed(random_state)
    # start with a real vector
    x_real = torch.randn(32, 40, 64, 64, 2).double()
    x_real[..., 1] = 0.
    # compute its FFT, which ensures hermitian symmetry
    x = Variable(fft_c2c_cpu(x_real))
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    # compute its IFFT in full
    x_back_full_cpu = fft_c2c_cpu(x.data, inverse=True)
    assert torch.max(torch.abs(x_back_full_cpu)[..., 1]) < 1e-7
    if test_cuda:
        x_gpu = Variable(x.data.clone()).cuda()
        # full IFFT
        x_back_full_gpu = ifft_c2c(x_gpu)
        x_back_full_gpu2 = x_back_full_gpu.data.cpu()
        assert torch.max(torch.abs(x_back_full_gpu2 - x_back_full_cpu)) < 1e-9
        assert torch.max(torch.abs(x_back_full_gpu2)[..., 1]) < 1e-9
        # quick IFFT
        x_back_quick_gpu = ifft_c2r(x_gpu)
        x_back_quick_gpu2 = x_back_quick_gpu.data.cpu()
        assert torch.max(torch.abs(x_back_quick_gpu2 - x_back_full_gpu2[..., 0])) < 1e-10
