from ...frontend.numpy_frontend import Scattering_numpy
from kymatio.scattering2d.core.scattering2d import scattering2d
from ..utils import compute_padding
from ..filter_bank import filter_bank

class Scattering2D_numpy(Scattering_numpy):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend=None):
        super(Scattering2D_numpy, self).__init__(J, shape, max_order=max_order)
        self.pre_pad, self.L, self.backend = pre_pad, L, backend
        self.build()

    def build(self):
        self.M, self.N = self.shape
        if not self.backend:
            from ..backend import torch_backend as backend
            self.backend = backend
        if 2 ** self.J > self.shape[0] or 2 ** self.J > self.shape[1]:
            raise RuntimeError('The smallest dimension should be larger than 2^J')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = backend.Pad(
            [(self.M_padded - self.M) // 2, (self.M_padded - self.M + 1) // 2, (self.N_padded - self.N) // 2,
             (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = backend.unpad
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    def scattering(self, input):
        return scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi, self.max_order, self.M_padded, self.N_padded)

    def __call__(self, x):
        return self.scattering(x)