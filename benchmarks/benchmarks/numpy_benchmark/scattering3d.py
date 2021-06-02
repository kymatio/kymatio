from kymatio.numpy import HarmonicScattering3D
import numpy as np

rng = np.random.RandomState(42)

class BenchmarkHarmonicScattering3D:
    param_names = ["sc_params"]
    timeout = 400  # in seconds

    params = [
        [
            {  # Small. 32x32x32, 2 scales, 2 harmonics
                "J": 2,
                "shape": (32, 32, 32),
                "L": 2,
                "batch_size": 4,
                "n_iter": 2
            },
            {  # Large. 128x128x128, 2 scales, 2 harmonics
                "J": 2,
                "shape": (128, 128, 128),
                "L": 2,
                "batch_size": 2,
                "n_iter": 2
            }
        ],
    ]

    def setup(self, sc_params):
        scattering = HarmonicScattering3D(J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        x = rng.randn(
            sc_params["batch_size"],
            sc_params["shape"][0],
            sc_params["shape"][1],
            sc_params["shape"][2]).astype("float32")
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params):
        HarmonicScattering3D(J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])

    def time_forward(self, sc_params):
        for i in range(sc_params["n_iter"]):
            y = self.scattering(self.x)
