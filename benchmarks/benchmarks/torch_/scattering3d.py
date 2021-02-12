from kymatio.torch import HarmonicScattering3D
import torch


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
    from kymatio.scattering3d.backend.torch_skcuda_backend import backend
    backends.append(backend)
    devices = ['cuda']

from kymatio.scattering3d.backend.torch_backend import backend
backends.append(backend)



class BenchmarkHarmonicScattering3D:
    param_names = ["sc_params", "backend", "devices"]
    timeout = 400  # in seconds

    params = [
        [
            { # Small. 32x32x32, 2 scales, 2 harmonics
                "J": 2,
                "shape": (32, 32, 32),
                "L": 2,
                "batch_size": 4
            },
            { # Large. 128x128x128, 2 scales, 2 harmonics
                "J": 2,
                "shape": (128, 128, 128),
                "L": 2,
                "batch_size": 2
            }
        ],
        backends,
        devices
    ]

    def setup(self, sc_params,  backend, device):
        scattering = HarmonicScattering3D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        bs = sc_params["batch_size"]
        if device == 'cuda':
            bs *= 2
        x = torch.randn(
            bs,
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).float()
        self.scattering = scattering
        self.x = x.to(device)

    def time_constructor(self, sc_params,  backend, device):
        if device == 'cuda':
            torch.cuda.synchronize()
        HarmonicScattering3D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
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

