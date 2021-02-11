from kymatio.torch import HarmonicScattering3D
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
    from kymatio.scattering1d.backend.torch_skcuda_backend import backend
    backends.append(backend)

from kymatio.scattering1d.backend.torch_backend import backend
backends.append(backend)

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


class BenchmarkHarmonicScattering3D:
    params = [
        [
            { # Small. 32x32x32, 2 scales, 2 harmonics
                "J": 2,
                "shape": (32, 32, 32),
                "L": 2,
            },
            { # Large. 128x128x128, 2 scales, 2 harmonics
                "J": 2,
                "shape": (128, 128, 128),
                "L": 2,
            }
        ],
        [
            1,
        ],
        backends,
        devices
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size, backend, device):
        scattering = HarmonicScattering3D(backend=backend, **sc_params)
        x = np.random.randn(
            batch_size,
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).astype("float32")
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size, backend, device):
        HarmonicScattering3D(backend=backend, **sc_params)

    def time_forward(self, sc_params, batch_size, backend, device):
        (self.scattering)=(self.scattering).to(device)
        (self.scattering).__call__(self.x.to(device))
