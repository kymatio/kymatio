from kymatio.torch import Scattering2D
import torch

torch.manual_seed(42)

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

BATCH_SIZE = 32
N_ITER = 2

if skcuda_available:
    from kymatio.scattering2d.backend.torch_skcuda_backend import backend
    backends.append(backend)
    devices = ['cuda']
    BATCH_SIZE = 256
    N_ITER = 10

from kymatio.scattering2d.backend.torch_backend import backend
backends.append(backend)



class BenchmarkScattering2D:
    timeout = 400 # in seconds
    param_names = ["sc_params", "backend", "device"]
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 4 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 8,
                "batch_size": BATCH_SIZE,
                "n_iter": N_ITER
            },
            { # ImageNet-like. 224x224, 3 scales, 4 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 8,
                "batch_size": BATCH_SIZE,
                "n_iter": N_ITER
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (64, 64),
                "L": 2,
                "batch_size": BATCH_SIZE//8,
                "n_iter": N_ITER
            },
        ],
        backends,
        devices
    ]

    def setup(self, sc_params, backend, device):
        n_channels = 3
        scattering = Scattering2D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        bs = sc_params["batch_size"]

        x = torch.randn(
            bs,
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).float().to(device)
        self.scattering = scattering.to(device)
        self.x = x.to(device)
        y = self.scattering(self.x) # always perform an initial forward before using it next because initialization
        # can take some time

    def time_constructor(self, sc_params,  backend, device):
        Scattering2D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])

    @torch.no_grad()
    def time_forward(self, sc_params, backend, device):
        if device == 'cuda':
            torch.cuda.synchronize()
        for i in range(sc_params["n_iter"]):
            y = self.scattering(self.x)
        if device == 'cuda':
            torch.cuda.synchronize()
