# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

import torch
from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering3d import scattering3d
from ..core.scattering3d_standard import scattering3d_standard
from .base_frontend import ScatteringHarmonicBase3D, ScatteringBase3D
from ..utils import compute_padding


class HarmonicScatteringTorch3D(ScatteringTorch, ScatteringHarmonicBase3D):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringHarmonicBase3D.__init__(self, J, shape, L, sigma_0, max_order,
                                  rotation_covariant, method, points,
                                  integral_powers, backend)

        self.build()

    def build(self):
        ScatteringHarmonicBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringHarmonicBase3D.build(self)
        ScatteringHarmonicBase3D.create_filters(self)

        self.register_filters()

    def register_filters(self):
        # transfer the filters from numpy to torch
        for k in range(len(self.filters)):
            filt = torch.zeros(self.filters[k].shape + (2,))
            filt[..., 0] = torch.from_numpy(self.filters[k].real).reshape(self.filters[k].shape)
            filt[..., 1] = torch.from_numpy(self.filters[k].imag).reshape(self.filters[k].shape)
            self.filters[k] = filt
            self.register_buffer('tensor' + str(k), self.filters[k])

        g = torch.zeros(self.gaussian_filters.shape + (2,))
        g[..., 0] = torch.from_numpy(self.gaussian_filters.real)
        self.gaussian_filters = g
        self.register_buffer('tensor_gaussian_filter', self.gaussian_filters)

    def scattering(self, input_array):
        if not torch.is_tensor(input_array):
            raise TypeError(
                'The input should be a torch.cuda.FloatTensor, '
                'a torch.FloatTensor or a torch.DoubleTensor.')

        if input_array.dim() < 3:
            raise RuntimeError('Input tensor must have at least three '
                               'dimensions.')

        if (input_array.shape[-1] != self.O or input_array.shape[-2] != self.N
            or input_array.shape[-3] != self.M):
            raise RuntimeError(
                'Tensor must be of spatial size (%i, %i, %i).' % (
                    self.M, self.N, self.O))

        input_array = input_array.contiguous()

        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]

        input_array = input_array.reshape((-1,) + signal_shape + (1,))


        buffer_dict = dict(self.named_buffers())
        for k in range(len(self.filters)):
            self.filters[k] = buffer_dict['tensor' + str(k)]

        methods = ['integral']
        if not self.method in methods:
            raise ValueError('method must be in {}'.format(methods))

        if self.method == 'integral': \
                self.averaging = lambda x: self.backend.compute_integrals(x, self.integral_powers)

        S = scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L,
                            J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)
        scattering_shape = S.shape[1:]

        S = S.reshape(batch_shape + scattering_shape)

        return S


HarmonicScatteringTorch3D._document()



class ScatteringTorch3D(ScatteringTorch, ScatteringBase3D):
    def __init__(self, J, shape, orientations="cartesian", max_order=2, pre_pad=False,
            backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, orientations, max_order, pre_pad, backend)
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

        self.register_filters()
        self.build()
    
    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for c, phi in self.phi.items():
            if not isinstance(c, int):
                continue

            self.phi[c] = self.register_single_filter(phi, n)
            n = n + 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if not isinstance(k, int):
                    continue

                self.psi[j][k] = self.register_single_filter(v, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        phis = self.phi
        for c, phi in phis.items():
            if not isinstance(c, int):
                continue

            phis[c] = self.load_single_filter(n, buffer_dict)
            n = n + 1

        psis = self.psi
        for j in range(len(psis)):
            for k, v in psis[j].items():
                if not isinstance(k, int):
                    continue

                psis[j][k] = self.load_single_filter(n, buffer_dict)
                n = n + 1

        return phis, psis

    def scattering(self, input):
        
        phi, psi = self.load_filters()

        batch_shape = input.shape[:-3]
        signal_shape = input.shape[-3:]

        input = input.reshape((-1,) + signal_shape)

        S = scattering3d_standard(input, self.pad, self.unpad, self.backend,
                self.J, len(self.orientations), phi, psi, self.max_order)

        scattering_shape = S.shape[-4:]
        S = S.reshape(batch_shape + scattering_shape)

        return S



__all__ = ['HarmonicScatteringTorch3D', 'ScatteringBase3D']
