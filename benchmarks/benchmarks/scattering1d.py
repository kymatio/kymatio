import torch
import kymatio.scattering1d.backend as backend
from kymatio import Scattering1D

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
        ],
        [
            32,
        ]
    ]
    param_names = ["sc_params", "batch_size"]

    def setup(self, sc_params, batch_size):
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

    def time_constructor(self, sc_params, batch_size):
        Scattering1D(**sc_params)

    def time_forward(self, sc_params, batch_size):
        (self.scattering).forward(self.x)
