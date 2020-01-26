import numpy as np
import torch
import kymatio.scattering2d.backend as backend
from kymatio import Scattering2D

from .common import SCATTERING_BENCHMARK_SIZE

class BenchmarkScattering2D:
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 8 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 8,
            },
            { # ImageNet-like. 256x256, 3 scales, 8 orientations
                "J": 3,
                "shape": (256, 256),
                "L": 8,
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (256, 256),
                "L": 2,
            },
        ]
    ]
    param_names = ["sc_params"]

    def setup(self, sc_params):
        signal_size = int(np.prod(sc_params["shape"]))
        batch_size = SCATTERING_BENCHMARK_SIZE // signal_size
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

    def time_constructor(self, sc_params):
        Scattering2D(**sc_params)

    def time_forward(self, sc_params):
        (self.scattering).forward(self.x)
