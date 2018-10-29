import torch

############################ TORCH BACKEND - FLOAT 32 -- FORWARD ##################
print('==> Testing Float32 with PyTorch and Torch backend')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

from scattering import Scattering2D as Scattering

scattering = Scattering(M=224, N=224, J=3, L=8)
x_data = torch.randn(256, 3, 224, 224).float()

torch.cuda.synchronize()
t_start = time.time()
for i in range(10):
    scattering(x_data)
torch.cuda.synchronize()
elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, 10))
print('Hz: %.2f [hz]' % (times / 10))