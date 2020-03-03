import torch
import kymatio.scattering2d.backend as backend
from kymatio import Scattering2D

class BenchmarkScattering2D:
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 8 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 8,
            },
            { # ImageNet-like. 224x224, 3 scales, 8 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 8,
            },
            { # A case with many scales (J=7) and few orientations (L=2)
                "J": 7,
                "shape": (224, 224),
                "L": 2,
            },
        ],
        [
            32,
        ]
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size):
        n_channels = 1
        scattering = Scattering2D(**sc_params)
        scattering.cpu()
        x = torch.randn(
            batch_size,
            n_channels,
            sc_params["shape"][0], sc_params["shape"][1],
            dtype=torch.float32)
        x.cpu()
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size):
        Scattering2D(**sc_params)

    def time_forward(self, sc_params, batch_size):
        (self.scattering).forward(self.x)
