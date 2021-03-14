# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

BACKEND_NAME = 'torch17_skcuda'


from .torch_skcuda_backend import backend
from .torch17_backend import backend as backend_torch

backend.fft = backend_torch.fft

backend.name = 'torch17_skcuda'
