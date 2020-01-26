import torch
import numpy as np
import kymatio.scattering1d.backend as backend
from kymatio import Scattering1D

from .common import SCATTERING_BENCHMARK_SIZE

class BenchmarkScattering1D:
    params = [
        [
            { # Typical of EEG. J=8, Q=1, N=1024
              # See Warrick et al. Physiological Measurement 2019
                "J": 8,
                "Q": 1,
                "shape": 1024,
            },
            { # Typical of speech.
              # See Andén and Mallat TASLP 2014
                "J": 8,
                "Q": 8,
                "shape": 4096,
            },
            { # Typical of music.
              # See Andén et al.
                "J": 16,
                "Q": 12,
                "shape": 131072,
            },
        ]
    ]
    param_names = ["sc_params"]

    def setup(self, sc_params):
        signal_size = int(np.prod(sc_params["shape"]))
        batch_size = SCATTERING_BENCHMARK_SIZE // signal_size
        n_channels = 1
        scattering = Scattering1D(**sc_params)
        scattering.cpu()
        x = torch.randn(
            batch_size,
            n_channels,
            sc_params["shape"],
            dtype=torch.float32)
        x.cpu()
        self.scattering = scattering
        self.x = x

    def time_constructor(self, sc_params):
        Scattering1D(**sc_params)

    def time_forward(self, sc_params):
        (self.scattering).forward(self.x)
