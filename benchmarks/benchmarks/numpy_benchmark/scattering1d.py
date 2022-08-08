from kymatio.numpy import Scattering1D
import numpy as np

rng = np.random.RandomState(42)

class BenchmarkScattering1D:
    param_names = ["sc_params"]
    timeout = 400  # in seconds

    params = [
        [
            {  # Typical of EEG. J=8, Q=1, N=1024
                # See Warrick et al. Physiological Measurement 2019
                "J": 8,
                "Q": 1,
                "shape": 1024,
                "batch_size": 32,
                "n_iter": 2
            },
            {  # Typical of speech.
                # See Andén and Mallat TASLP 2014
                "J": 8,
                "Q": 8,
                "shape": 4096,
                "batch_size": 32,
                "n_iter": 2
            },
            {  # Typical of music.
                # See Andén et al.
                "J": 13,
                "Q": 12,
                "shape": 65536,
                "batch_size": 32,
                "n_iter": 2
            },
        ],
    ]

    def setup(self, sc_params):
        n_channels = 1
        scattering = Scattering1D(J=sc_params["J"], shape=sc_params["shape"], Q=sc_params["Q"])
        x = rng.randn(
            sc_params["batch_size"],
            n_channels,
            sc_params["shape"]).astype("float32")
        self.scattering = scattering
        self.x = x
        y = self.scattering(self.x)  # always perform an initial forward before using it next because initialization
        # can take some time

    def time_constructor(self, sc_params):
        Scattering1D(J=sc_params["J"], shape=sc_params["shape"], Q=sc_params["Q"])

    def time_forward(self, sc_params):
        for i in range(sc_params["n_iter"]):
            y = self.scattering(self.x)
