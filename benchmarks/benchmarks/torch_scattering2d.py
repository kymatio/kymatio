from kymatio.torch import Scattering2D
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
    from kymatio.scattering2d.backend.torch_skcuda_backend import backend
    backends.append(backend)
    devices = ['cuda']

from kymatio.scattering2d.backend.torch_backend import backend
backends.append(backend)




class BenchmarkScattering2D:
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 4 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 8,
                "batch_size":128
            },
            { # ImageNet-like. 224x224, 3 scales, 4 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 8,
                "batch_size": 256
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (64, 64),
                "L": 2,
                "batch_size": 4

            },
        ],
        backends,
        devices
    ]
    param_names = ["sc_params", "backend", "device"]

    def setup(self, sc_params, backend, device):
        n_channels = 3
        scattering = Scattering2D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        x = torch.randn(
            sc_params["batch_size"],
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).float().to(device)
        self.scattering = scattering.to(device)
        self.x = x

    def time_constructor(self, sc_params,  backend, device):
        Scattering2D(backend=backend, **sc_params)

    def time_forward(self, sc_params, backend, device):
        torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
        for i in range(10):
            print(i)
            y =self.scattering(self.x)

