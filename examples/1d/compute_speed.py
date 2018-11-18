"""
1D scattering transform benchmark
=================================
We compute scattering transforms for signals of length `T = 2**16`, with scale
`J = 10` and `Q = 8` wavelets per octave. The signals are stacked into batches
of size `batch_size = 64` and the transform is computed `10` times to get an
average running time.
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

import kymatio.scattering1d.backend as backend

###############################################################################
# Finally, we import the `Scattering1D` class that computes the scattering
# transform.
from kymatio import Scattering1D

###############################################################################
# Benchmark setup
# --------------------
# First, we set up some basic parameters, the signal length `T`, the number of
# wavelets per octave `Q`, and the averaging scale, `2**J`. For a sampling rate
# of 11025 Hz, `T = 2**16` corresponds to about 6 seconds of audio, and an
# averaging scale of `2**10` is about 100 milliseconds, both of which are
# typical values for these parameters in audio applications. For `Q`, 8
# wavelets per octave ensures that we are able to resolve isolated sinusoids
# without sacrificing too much temporal resolution.

T = 2**16
J = 10
Q = 8

###############################################################################
# To squeeze the maximum performance out of the implementation, we apply it to
# a batch of `64` signals. Larger batch sizes do not yield increased efficiency,
# but smaller values increases the influence of overhead on the running time.

batch_size = 64

###############################################################################
# We repeat the benchmark `10` times and compute the average running time to
# get a reasonable estimate.

times = 10

###############################################################################
# Determine which devices (CPU or GPU) that are supported by the current
# backend.

devices = []
if backend.NAME == 'torch':
    devices.append('cpu')
if backend.NAME == 'torch' and torch.cuda.is_available():
    devices.append('gpu')
if backend.NAME == 'skcuda' and torch.cuda.is_available():
    devices.append('gpu')

###############################################################################
# Create the `Scattering1D` object using the given parameters and generate
# some compatible test data with the specified batch size.

scattering = Scattering1D(T, J, Q)

x = torch.randn(batch_size, 1, T, dtype=torch.float32)

###############################################################################
# Run the benchmark
# -----------------
# For each device, we need to convert the `scattering` object and the Tensor
# `x` to the appropriate type, invoke `times` calls to the `scattering.forward`
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
#   Elapsed time: 27.158231 [s / 10 evals], avg: 2.72 (s/batch)
#   ==> Testing Float32 with torch backend, on GPU, forward
#   Elapsed time: 8.083082 [s / 10 evals], avg: 0.81 (s/batch)
