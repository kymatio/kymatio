from kymatio.numpy import Scattering2D
import numpy as np

rng = np.random.RandomState(42)

class BenchmarkScattering2D:
    timeout = 400 # in seconds
    param_names = ["sc_params"]
    params = [
        [
            { # MNIST-like. 32x32, 2 scales, 4 orientations
                "J": 2,
                "shape": (32, 32),
                "L": 8,
                "batch_size": 32,
                "n_iter": 2
            },
            { # ImageNet-like. 224x224, 3 scales, 4 orientations
                "J": 3,
                "shape": (224, 224),
                "L": 8,
                "batch_size": 32,
                "n_iter": 2
            },
            { # A case with many scales (J=6) and few orientations (L=2)
                "J": 6,
                "shape": (64, 64),
                "L": 2,
                "batch_size": 4,
                "n_iter": 2
            },
        ],
    ]

    def setup(self, sc_params):
        n_channels = 3
        scattering = Scattering2D(J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])
        x = rng.randn(
            sc_params["batch_size"],
            n_channels,
            sc_params["shape"][0],
            sc_params["shape"][1]).astype("float32")
        self.scattering = scattering
        self.x = x
        y = self.scattering(self.x) # always perform an initial forward before using it next because initialization
        # can take some time

    def time_constructor(self, sc_params):
        Scattering2D(J=sc_params["J"], shape=sc_params["shape"], L=sc_params["L"])

    def time_forward(self, sc_params):
        for i in range(sc_params["n_iter"]):
            y = self.scattering(self.x)
