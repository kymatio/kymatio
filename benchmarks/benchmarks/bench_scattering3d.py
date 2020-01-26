import numpy as np
import torch
import kymatio.scattering3d.backend as backend
from kymatio import HarmonicScattering3D

from .common import SCATTERING_BENCHMARK_SIZE

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
            },
            { # A case with many scales (J=6) and few harmonics (L=1)
                "J": 6,
                "shape": (128, 128, 128),
                "L": 1,
            },
            { # A case with few scales (J=2) and many harmonics (L=6)
                "J": 2,
                "shape": (32, 32, 32),
                "L": 4,
            }
        ]
    ]
    param_names = ["sc_params"]

    def setup(self, sc_params):
        scattering = HarmonicScattering3D(**sc_params)
        signal_size = int(np.prod(sc_params["shape"]))
        batch_size = SCATTERING_BENCHMARK_SIZE // signal_size
        scattering.cpu()
        x = torch.randn(
            batch_size,
            sc_params["shape"][0], sc_params["shape"][1], sc_params["shape"][2],
            dtype=torch.float32)
        x.cpu()
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params):
        HarmonicScattering3D(**sc_params)

    def time_forward(self, sc_params):
        (self.scattering).forward(self.x)
