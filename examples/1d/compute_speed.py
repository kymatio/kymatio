import torch
import time
import scattering.scattering1d.backend as backend
from scattering import Scattering1D

T = 2**16
Q = 8
J = 10

batch_size = 64

times = 10

scattering = Scattering1D(T, J, Q)

x_data = torch.randn(batch_size, 1, T, dtype=torch.float32)

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
            x_data = x_data.cpu()

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
