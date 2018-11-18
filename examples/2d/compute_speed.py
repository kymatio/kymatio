"""
2D scattering transform benchmark
=================================
We compute scattering transforms for images of size `256`-by-`256` with
averaging scale `2**3 = 8` and `L = 8` angular directions. The images are
stacked into batches of size `batch_size = 128` and the transforms are
computed `10` times to get an average running time.
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

import kymatio.scattering2d.backend as backend

###############################################################################
# Finally, we import the `Scattering2D` class that computes the scattering
# transform.

from kymatio import Scattering2D

###############################################################################
# Benchmark setup
# --------------------
# First, we set up some basic parameters: the image width `M` and height `N`,
# the averaging scale, `2**J`, and the number of angular directions `L`.
# Here, we consider square images of size `256` with an averaging scale
# `2**3 = 8` and `L = 8` angular directions. These are all typical parameter
# for scattering transforms of natural images.

M = 256
N = 256
J = 3
L = 8

###############################################################################
# To squeeze the maximum performance out of the implementation, we apply it to
# a batch of `128` images.

batch_size = 128

###############################################################################
# We repeat the benchmark `10` times and compute the average running time to
# get a reasonable estimate.

times = 10

###############################################################################
# Determine which devices (CPU or GPU) that are supported by the current
# backend.

if backend.NAME == 'torch':
    devices = ['cpu', 'gpu']
elif backend.NAME == 'skcuda':
    devices = ['gpu']

###############################################################################
# Create the `Scattering2D` object using the given parameters and generate
# some compatible test data with the specified batch size. The number of
# channels in the test data here is set to `3`, corresponding to the three
# colors channels in an RGB image.

scattering = Scattering2D(M, N, J, L)

x = torch.randn(batch_size, 3, M, N, dtype=torch.float32)

###############################################################################
# Run the benchmark
# -----------------
# For each device, we need to convert the `scattering` object and the Tensor
# `x` to the appropriate type, invoke `times` calls to `scattering.forward`
# and print the running times. Before the timer starts, we add an extra
# `scattering.forward` call to ensure any first-time overhead, such as memory
# allocation and CUDA kernel compilation, is not counted. If the benchmark is
# running on the GPU, we also need to call `torch.cuda.synchronize()` before
# and after the benchmark to make sure that all CUDA kernels have finished
# executing.

for device in devices:
    fmt_str = '==> Testing Float32 with {} backend, on {}, forward'
    print(fmt_str.format(backend.NAME, device.upper()))

    if device == 'gpu':
        scattering.cuda()
        x = x.cuda()
    else:
        scattering.cpu()
        x = x.cpu()

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
#   Elapsed time: 624.910853 [s / 10 evals], avg: 62.49 (s/batch)
#   ==> Testing Float32 with torch backend, on GPU, forward
#   Elapsed time: 130.580992 [s / 10 evals], avg: 13.06 (s/batch)
