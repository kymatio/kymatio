import torch
from kymatio.torch import Scattering1D



backends = []

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']

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
    devices = ['cuda']

from kymatio.scattering1d.backend.torch_backend import backend
backends.append(backend)



class BenchmarkScattering1D:
    param_names = ["sc_params", "backend", "devices"]
    timeout = 400  # in seconds

    params = [
        [
            {  # Typical of EEG. J=8, Q=1, N=1024
                # See Warrick et al. Physiological Measurement 2019
                "J": 8,
                "Q": 1,
                "shape": 1024,
                "batch_size": 256
            },
            {  # Typical of speech.
                # See Andén and Mallat TASLP 2014
                "J": 8,
                "Q": 8,
                "shape": 4096,
                "batch_size": 256
            },
            {  # Typical of music.
                # See Andén et al.
                "J": 13,
                "Q": 12,
                "shape": 65536,
                "batch_size": 256
            },
        ],
        backends,
        devices
    ]
    def setup(self, sc_params,  backend, device):
        n_channels = 1
        scattering = Scattering1D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], Q=sc_params["Q"])
        x = torch.randn(
            sc_params["batch_size"],
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).float().to(device)
        self.scattering = scattering.to(device)
        self.x = x
        y = self.scattering(self.x) # always perform an initial forward before using it next because initialization
        # can take some time

    def time_constructor(self, sc_params,  backend, device):
        if device == 'cuda':
            torch.cuda.synchronize()
        Scattering1D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], Q=sc_params["Q"])
        if device == 'cuda':
            torch.cuda.synchronize()

    def time_forward(self, sc_params, backend, device):
        n_iter = 2
        if device=='cuda':
            torch.cuda.synchronize()
            n_iter = 10
        for i in range(n_iter):
            y =self.scattering(self.x)
        if device == 'cuda':
            torch.cuda.synchronize()