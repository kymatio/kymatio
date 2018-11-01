"""
Benchmark the speed of the 3D scattering transform
==================================================
We compute scattering transforms for signals of length `T = 2**16`, with scale
`J = 10` and `Q = 8` wavelets per octave. The signals are stacked into batches
of size `batch_size = 64` and the transform is computed `10` times to get an
average running time.
"""

import torch
import time
from scattering import Scattering3D

J, L = 2, 2
integral_powers = [1., 2.]

###############################################################################
# Parameters for the benchmark
# ----------------------------

batch_size, M, N, O = 8, 128, 128, 128

times = 10

###############################################################################
# Set up the scattering object and the test data
# ----------------------------------------------

scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=1.)
x_data = torch.randn(batch_size, M, N, O).float()

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
            x_data = torch.randn(1, M, N, O).float()
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