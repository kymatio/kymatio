import torch
import time
from scattering import Scattering3D


batch_size, M, N, O = 8, 128, 128, 128
J, L = 2, 2
integral_powers = [1., 2.]

x_data = torch.randn(batch_size, M, N, O).float()
scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=1.)

print('Input signal of size {}'.format(x_data.size()))
print('Scattering transform, L={}, J={}, integral_powers={}'.format(L, J, integral_powers))

print('Testing 3D scattering on CPU Float32 with PyTorch and Torch backend')
t_start = time.time()
for i in range(10):
    scattering(x_data, order_2=True, method='integral', integral_powers=integral_powers)
elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, 10))

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
x_data_gpu = x_data.cuda()

torch.cuda.synchronize()

print('Testing 3D scattering on GPU Float32 with PyTorch and Torch backend')
t_start = time.time()
for i in range(10):
    scattering(x_data_gpu, order_2=True, method='integral', integral_powers=integral_powers)
torch.cuda.synchronize()
elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, 10))
