import torch
import time
import scattering.scattering2d.backend as backend
from scattering import Scattering2D as Scattering

scattering = Scattering(M=256, N=256, J=3, L=8)
x_data = torch.randn(128, 3, 256, 256)

times = 10
elapsed_time = -1

if backend.NAME == 'skcuda':
    ############################ SKCUDA BACKEND GPU - FLOAT 32 -- FORWARD ##################
    print('==> Testing Float32 with Skcuda backend, on GPU, forward')
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    scattering.cuda()
    x_data = x_data.cuda().float()

    # one first pass is perform to precompile the kernels and allocte memory...
    scattering(x_data)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        scattering(x_data)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start



if backend.NAME == 'torch' and not torch.cuda.is_available():
    ############## TORCH BACKEND CPU - FLOAT 32 -- FORWARD ##################
    print('==> Testing Float32 with Torch backend, on CPU, forward')
    from scattering import Scattering2D as Scattering

    scattering.cpu()
    x_data = x_data.cpu().float()

    # one first pass to allocate memory..
    scattering(x_data)

    t_start = time.time()
    for i in range(10):
        scattering(x_data)
    elapsed_time = time.time() - t_start


if backend.NAME == 'torch' and torch.cuda.is_available():
    ############## TORCH BACKEND GPU - FLOAT 32 -- FORWARD ##################
    print('==> Testing Float32 with Torch backend, on GPU, forward')
    from scattering import Scattering2D as Scattering

    scattering.cuda()
    x_data = x_data.cuda().float()

    # one first pass is perform to precompile the kernels and allocte memory...
    scattering(x_data)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        scattering(x_data)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals], avg: %.2f (s/batch)' % (elapsed_time, times, elapsed_time/times))



