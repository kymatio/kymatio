from ...frontend.numpy_frontend import Scattering_numpy
from ..utils import compute_padding

class Scattering2D(Scattering_numpy):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False):
        super(Scattering2D, self).__init__(J, shape, max_order=max_order)
        self.pre_pad, self.L = pre_pad, L
        self.build()

    def build(self):
        self.M, self.N = self.shape
        if 2 ** self.J > self.shape[0] or 2 ** self.J > self.shape[1]:
            raise RuntimeError('The smallest dimension should be larger than 2^J')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = backend.Pad(
            [(self.M_padded - self.M) // 2, (self.M_padded - self.M + 1) // 2, (self.N_padded - self.N) // 2,
             (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = backend.unpad
        self.create_and_register_filters()

    def scattering(self, input):
        return scattering2d(input, self.J, self.L, self.subsample_fourier, self.pad, self.modulus, fft, cdgmm, unpad, self.phi, self.psi, self.max_order, self.M_padded, self.N_padded)