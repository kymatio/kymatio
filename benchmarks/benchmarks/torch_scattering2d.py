from kymatio.torch import Scattering2D
import numpy as np
import torch


backends = []

skcuda_available = False
try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        skcuda_available = True
except:
    Warning('torch_skcuda backend not available.')

if skcuda_available:
    from kymatio.scattering2d.backend.torch_skcuda_backend import backend
    backends.append(backend)

from kymatio.scattering2d.backend.torch_backend import backend
backends.append(backend)

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


class BenchmarkScattering2D:
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 4 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 4,
            },
            { # ImageNet-like. 224x224, 3 scales, 4 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 4,
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (224, 224),
                "L": 2,
            },
        ],
        [
            1,
        ],
        backends,
        devices
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size, backend, device):
        n_channels = 1
        scattering = Scattering2D(backend=backend, **sc_params)
        x = torch.randn(
            batch_size,
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).float()
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size, backend, device):
        Scattering2D(backend=backend, **sc_params)

    def time_forward(self, sc_params, batch_size, backend, device):
        (self.scattering)=(self.scattering).to(device)
        (self.scattering).__call__(self.x.to(device))
