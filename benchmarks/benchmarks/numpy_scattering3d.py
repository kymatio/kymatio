from kymatio.numpy import HarmonicScattering3D
import numpy as np

class BenchmarkHarmonicScattering3D:
    param_names = ["sc_params"]
    timeout = 400  # in seconds

    params = [
        [
            {  # Small. 32x32x32, 2 scales, 2 harmonics
                "J": 2,
                "shape": (32, 32, 32),
                "L": 2,
                "batch_size": 16
            },
            {  # Large. 128x128x128, 2 scales, 2 harmonics
                "J": 2,
                "shape": (128, 128, 128),
                "L": 2,
                "batch_size": 16
            }
        ],
    ]

    def setup(self, sc_params):
        scattering = HarmonicScattering3D(**sc_params)
        x = np.random.randn(
            sc_params["batch_size"],
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).astype("float32")
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params):
        HarmonicScattering3D(**sc_params)

    def time_forward(self, sc_params):
        n_iter = 2
        for i in range(n_iter):
            y = self.scattering(self.x)