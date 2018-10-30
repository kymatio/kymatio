import torch

import scattering.scattering2d.backend as backend
from scattering import Scattering2D as Scattering


if backend.NAME == 'skcuda':
    ############################ TORCH BACKEND - FLOAT 32 -- FORWARD ##################
    print('==> Testing Float32 with PyTorch and Torch backend')
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    scattering = Scattering(M=224, N=224, J=3, L=8).cuda()
    x_data = torch.randn(256, 3, 224, 224).cuda().float()

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        scattering(x_data)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, 10))
    print('Hz: %.2f [hz]' % (times / 10))




    print('skcuda backend not installed... passing...\n')


if backend.NAME == 'torch':
    ############## FIRST CPU TEST, TORCH BACKEND - FLOAT 32 -- FORWARD ##################
    print('==> Testing Float32 with PyTorch and Torch backend, on CPU, forward')
    from scattering import Scattering2D as Scattering

    scattering = Scattering(M=224, N=224, J=3, L=8).cpu()
    x_data = torch.randn(256, 3, 224, 224).float()

    t_start = time.time()
    for i in range(10):
        scattering(x_data)
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, 10))
    print('Hz: %.2f [hz]' % (times / 10))



