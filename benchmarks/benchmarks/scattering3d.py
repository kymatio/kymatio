from kymatio.numpy import HarmonicScattering3D
import numpy as np

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
        ]
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size):
        scattering = HarmonicScattering3D(**sc_params)
        x = np.random.randn(
            batch_size,
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).astype("float32")
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size):
        HarmonicScattering3D(**sc_params)

    def time_forward(self, sc_params, batch_size):
        (self.scattering).__call__(self.x)
