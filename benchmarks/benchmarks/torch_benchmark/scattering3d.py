from kymatio.torch import HarmonicScattering3D
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

BATCH_SIZE = 4
N_ITER = 2

if skcuda_available:
    from kymatio.scattering3d.backend.torch_skcuda_backend import backend
    backends.append(backend)
    devices = ['cuda']
    BATCH_SIZE = 8
    N_ITER = 10

from kymatio.scattering3d.backend.torch_backend import backend
backends.append(backend)



class BenchmarkHarmonicScattering3D:
    param_names = ["sc_params", "backend", "device"]
    timeout = 400  # in seconds

    params = [
        [
            { # Small. 32x32x32, 2 scales, 2 harmonics
                "J": 2,
                "shape": (32, 32, 32),
                "L": 2,
                "batch_size": BATCH_SIZE,
                "n_iter": N_ITER
            },
            { # Large. 128x128x128, 2 scales, 2 harmonics
                "J": 2,
                "shape": (128, 128, 128),
                "L": 2,
                "batch_size": BATCH_SIZE//2,
                "n_iter": N_ITER
            }
        ],
        backends,
        devices
    ]

    def setup(self, sc_params,  backend, device):
        scattering = HarmonicScattering3D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        bs = sc_params["batch_size"]
        x = torch.randn(
            bs,
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).float()
        self.scattering = scattering.to(device)
        self.x = x.to(device)

    def time_constructor(self, sc_params,  backend, device):
        HarmonicScattering3D(backend=backend, J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])

    @torch.no_grad()
    def time_forward(self, sc_params, backend, device):
        if device == 'cuda':
            torch.cuda.synchronize()
        for i in range(sc_params["n_iter"]):
            y = self.scattering(self.x)
        if device == 'cuda':
            torch.cuda.synchronize()
