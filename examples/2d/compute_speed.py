"""
Benchmark the speed of the 1D scattering transform
==================================================
We compute scattering transforms for signals of length `T = 2**16`, with scale
`J = 10` and `Q = 8` wavelets per octave. The signals are stacked into batches
of size `batch_size = 64` and the transform is computed `10` times to get an
average running time.
"""

import torch
import time
import scattering.scattering2d.backend as backend
from scattering import Scattering2D as Scattering

###############################################################################
# Parameters for the benchmark
# ----------------------------

M = 256
N = 256

batch_size = 128

J = 3
L = 8

times = 10

###############################################################################
# Set up the scattering object and the test data
# ----------------------------------------------

scattering = Scattering(M=M, N=N, J=J, L=L)

x_data = torch.randn(batch_size, 3, M, N, dtype=torch.float32)

###############################################################################
# Benchmark the PyTorch backend
# -----------------------------
# If we're using the this backend, compute scattering transforms both on CPU
# and GPU so that we can compare performance.

if backend.NAME == 'torch':
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('gpu')

    for device in devices:
        fmt_str = '==> Testing Float32 with Torch backend, on {}, forward'
        print(fmt_str.format(device.upper()))

        if device == 'gpu':
            scattering.cuda()
            x_data = x_data.cuda()
        else:
            scattering.cpu()
            x_data = torch.randn(5, 3, M, N, dtype=torch.float32)
            x_data = x_data.cpu()

        # One pass because the first forward is always slower
        scattering(x_data)

        if device == 'gpu':
            torch.cuda.synchronize()

        t_start = time.time()
        for _ in range(times):
            scattering(x_data)

        if device == 'gpu':
            torch.cuda.synchronize()

        t_elapsed = time.time() - t_start

        fmt_str = 'Elapsed time: {:2f} [s / {:d} evals], avg: {:.2f} (s/batch)'
        print(fmt_str.format(t_elapsed, times, t_elapsed/times))