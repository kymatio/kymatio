"""
3D scattering transform benchmark (PyTorch)
===========================================
We compute scattering transforms for volume maps of size `128`-by-`128`-by-
`128`, with averaging scale `2**2 = 4` and maximum spherical harmonic
order `L = 2`. The volumes are stacked into batches of size `batch_size = 8`
and the transforms are computed `10` times to get an average running time.
"""

###############################################################################
# Preliminaries
# -------------
# Since kymatio handles PyTorch arrays, we first import `torch`.

import torch

###############################################################################
# To measure the running time of the implementation, we use the `time` package.

import time

###############################################################################
# The performance of the implementation depends on which "backend" is used. We
# therefore want to report the name of the backend when presenting the results.
# Certain backends are also GPU-only, we we want to detect that before running
# the benchmark.

backends = []

try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        from kymatio.scattering3d.backend.torch_skcuda_backend import backend
        backends.append(backend)
except:
    pass

try:
    from kymatio.scattering3d.backend.torch_backend import backend
    backends.append(backend)
except:
    pass

###############################################################################
# Finally, we import the `HarmonicScattering3D` class that computes the scattering
# transform.

from kymatio import HarmonicScattering3D
###############################################################################
# Benchmark setup
# --------------------
# First, we set up some basic parameters: the volume width `M`, height `N`,
# and depth 'O', the maximum number of the spherical harmonics `L`, and the
# maximum scale `2**J`. Here, we consider cubic volumes of size `128`, with
# a maximum scale of `2**2 = 4` and maximum spherical harmonic order of `2`.

M, N, O = 128, 128, 128
J = 2
L = 2

integral_powers = [1., 2.]
sigma_0 = 1

###############################################################################
# To squeeze the maximum performance out of the implementation, we apply it to
# a batch of `8` volumes. Larger batch sizes do not yield increased efficiency,
# but smaller values increases the influence of overhead on the running time.

batch_size = 8

###############################################################################
# We repeat the benchmark `10` times and compute the average running time to
# get a reasonable estimate.

times = 10

###############################################################################
# Determine which devices (CPU or GPU) that are supported by the current
# backend.

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']

###############################################################################
# Set up the scattering object and the test data
# ----------------------------------------------

###############################################################################
# Create the `HarmonicScattering3D` object using the given parameters and generate
# some compatible test data with the specified batch size.

x = torch.randn(batch_size, M, N, O, dtype=torch.float32)

###############################################################################
# Run the benchmark
# -----------------
# For each device, we need to convert the Tensor `x` to the appropriate type,
# invoke `times` calls to `scattering.forward` and print the running times.
# Before the timer starts, we add an extra `scattering.forward` call to ensure
# any first-time overhead, such as memory allocation and CUDA kernel
# compilation, is not counted. If the benchmark is running on the GPU, we also
# need to call `torch.cuda.synchronize()` before and after the benchmark to
# make sure that all CUDA kernels have finished executing.

for backend in backends:
    scattering = HarmonicScattering3D(J, shape=(M, N, O), L=L, sigma_0=sigma_0, backend=backend, frontend='torch')
    for device in devices:
        fmt_str = '==> Testing Float32 with {} backend, on {}, forward'
        print(fmt_str.format(backend.name, device.upper()))

        if not(device == 'cpu' and backend.name == 'torch_skcuda'):
            x, scattering = x.to(device), scattering.to(device)
        else:
            continue

        scattering.method = 'integral'
        scattering.integral_powers = integral_powers

        scattering.forward(x)

        if device == 'gpu':
            torch.cuda.synchronize()

        t_start = time.time()
        for _ in range(times):
            scattering.forward(x)

        if device == 'gpu':
            torch.cuda.synchronize()

        t_elapsed = time.time() - t_start

        fmt_str = 'Elapsed time: {:2f} [s / {:d} evals], avg: {:.2f} (s/batch)'
        print(fmt_str.format(t_elapsed, times, t_elapsed/times))

###############################################################################
# The resulting output should be something like
#
# .. code-block:: text
#
#   ==> Testing Float32 with torch backend, on CPU, forward
#   Elapsed time: 109.739110 [s / 10 evals], avg: 10.97 (s/batch)
#   ==> Testing Float32 with torch backend, on GPU, forward
#   Elapsed time: 60.476041 [s / 10 evals], avg: 6.05 (s/batch)
