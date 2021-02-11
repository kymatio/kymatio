from kymatio.torch import Scattering1D
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

params = [
    [
        { # Typical of EEG. J=8, Q=1, N=1024
            # See Warrick et al. Physiological Measurement 2019
            "J": 8,
            "Q": 1,
            "shape": 1024,
        },
        { # Typical of speech.
            # See Andén and Mallat TASLP 2014
            "J": 8,
            "Q": 8,
            "shape": 4096,
        },
        { # Typical of music.
            # See Andén et al.
            "J": 13,
            "Q": 12,
            "shape": 65536,
        },
    ],
    [
        1,
    ],
    backends,
    devices
]
class BenchmarkScattering1D:
    param_names = ["sc_params", "batch_size", "backend", "devices"]

    def setup(self, sc_params, batch_size, backend, device):
        n_channels = 1
        scattering = Scattering1D(backend=backend, **sc_params)
        x = torch.randn(
            batch_size,
            n_channels,
            sc_params["shape"]).float()
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size, backend, device):
        Scattering1D(backend=backend,**sc_params)

    def time_forward(self, sc_params, batch_size, backend,device):
        (self.scattering)=(self.scattering).to(device)
        (self.scattering).__call__(self.x.to(device))
