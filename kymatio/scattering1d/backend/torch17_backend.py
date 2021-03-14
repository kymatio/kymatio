# Authors: Edouard Oyallon, Joakim Anden, Mathieu Andreux

import torch
import torch.nn.functional as F

from collections import namedtuple

BACKEND_NAME = 'torch'

from ...backend.torch_backend import type_checks
from ...backend.base_backend import FFT
from .torch_backend import backend


fft = FFT(lambda x: torch.fft(x, 1, normalized=False),
    lambda x: torch.ifft(x, 1, normalized=False),
    lambda x: torch.irfft(x, 1, normalized=False, onesided=False),
    type_checks)


backend.fft = fft
