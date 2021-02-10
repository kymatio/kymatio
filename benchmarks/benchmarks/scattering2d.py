from kymatio.numpy import Scattering2D
import numpy as np

class BenchmarkScattering2D:
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 4 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 4,
            },
            { # ImageNet-like. 224x224, 3 scales, 4 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 4,
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (224, 224),
                "L": 2,
            },
        ],
        [
            1,
        ]
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size):
        n_channels = 1
        scattering = Scattering2D(**sc_params)
        x = np.random.randn(
            batch_size,
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).astype("float32")
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params, batch_size):
        Scattering2D(**sc_params)

    def time_forward(self, sc_params, batch_size):
        (self.scattering).__call__(self.x)
