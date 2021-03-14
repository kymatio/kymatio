# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import torch

BACKEND_NAME = 'torch17'

from ...backend.torch_backend import type_checks
from ...backend.base_backend import FFT
from .torch_backend import backend



fft = FFT(lambda x: torch.fft(x, 2, normalized=False),
          lambda x: torch.ifft(x, 2, normalized=False),
          lambda x: torch.irfft(x, 2, normalized=False, onesided=False),
          type_checks)


backend.fft = fft
backend.name = 'torch17'
