# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

import torch
from ...frontend.torch_frontend import ScatteringTorch
#from ..core.scattering3d import scattering3d
from ..core.scattering3d_standard import scattering3d
import numpy as np

from .base_frontend import ScatteringBase3D


class HarmonicScatteringTorch3D(ScatteringTorch, ScatteringBase3D):
    """3D Solid Harmonic scattering .
    This class implements solid harmonic scattering on an input 3D image.
    For details see https://arxiv.org/abs/1805.00571.
    Instantiates and initializes a 3d solid harmonic scattering object.
    Parameters
    ----------
    J: int
        number of scales
    shape: tuple of int
        shape (M, N, O) of the input signal
    L: int, optional
        Number of l values. Defaults to 3.
    sigma_0: float, optional
        Bandwidth of mother wavelet. Defaults to 1.
    max_order: int, optional
        The maximum order of scattering coefficients to compute. Must be
        either 1 or 2. Defaults to 2.
    rotation_covariant: bool, optional
        if set to True the first order moduli take the form:
        $\\sqrt{\\sum_m (x \\star \\psi_{j,l,m})^2)}$
        if set to False the first order moduli take the form:
        $x \\star \\psi_{j,l,m}$
        The second order moduli change analogously
        Defaut: True
    method: string, optional
        specifies the method for obtaining scattering coefficients
        ("standard","local","integral"). Default: "standard"
    points: array-like, optional
        List of locations in which to sample wavelet moduli. Used when
        method == 'local'
    integral_powers: array-like
        List of exponents to the power of which moduli are raised before
        integration. Used with method == 'standard', method == 'integral'
    """
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='standard', points=None,
                 integral_powers=(0.5, 1., 2.), backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order, rotation_covariant, method, points,
                 integral_powers, backend)
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

        self.register_filters()
        self.J = J
        self.shape = shape
        self.L = L
        self.sigma_0 = sigma_0

        self.max_order = max_order
        self.rotation_covariant = rotation_covariant
        self.method = method
        self.points = points
        self.integral_powers = integral_powers
        self.backend = backend

        self.build()

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
        buffer_dict = dict(self.named_buffers())
        for k in range(len(self.filters)):
            self.filters[k] = buffer_dict['tensor' + str(k)]

        methods = ['standard', 'local', 'integral']
        if (not self.method in methods):
            raise (ValueError('method must be in {}'.format(methods)))
        if self.method == 'integral': \
                self.averaging = lambda x, j: self.backend.compute_integrals(self.backend.fft(x, inverse=True)[..., 0],
                                                                             self.integral_powers)
        elif self.method == 'local':
            self.averaging = lambda x, j: \
                self.backend._compute_local_scattering_coefs(x,
                                                             self.tensor_gaussian_filter, j, self.points)
        elif self.method == 'standard':
            self.averaging = lambda x, j: \
                self.backend._compute_standard_scattering_coefs(x,
                                                                self.tensor_gaussian_filter, self.J,
                                                                self.backend.subsample)

        return scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L,
                            J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)

    def forward(self, input_array):
        """
        The forward pass of 3D solid harmonic scattering
        Parameters
        ----------
        input_array: torch tensor
            input of size (batchsize, M, N, O)
        Returns
        -------
        output: tuple | torch tensor
            if max_order is 1 it returns a torch tensor with the
            first order scattering coefficients
            if max_order is 2 it returns a torch tensor with the
            first and second order scattering coefficients,
            concatenated along the feature axis
        """
        if not torch.is_tensor(input_array):
            raise (TypeError(
                'The input should be a torch.cuda.FloatTensor, '
                'a torch.FloatTensor or a torch.DoubleTensor'))

        if not input_array.is_contiguous():
            input_array = input_array.contiguous()

        if ((input_array.shape[-1] != self.O or input_array.shape[-2] != self.N
             or input_array.shape[-3] != self.M)):
            raise (RuntimeError(
                'Tensor must be of spatial size (%i,%i,%i)!' % (
                    self.M, self.N, self.O)))

        if (input_array.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]

        x = input_array.reshape((-1, 1) + signal_shape)
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)

        S = self.scattering(x)
        scattering_shape = S.shape[-4:]
        S = S.reshape(batch_shape + scattering_shape)
        return S

class ScatteringTorch3D(ScatteringTorch, ScatteringBase3D):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='standard', points=None,
                 integral_powers=(0.5, 1., 2.), backend='torch', pre_pad=False,
                 orientations='cartesian'):
        ScatteringTorch.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order, rotation_covariant, method, points,
                 integral_powers, backend, pre_pad, orientations)
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

    def scattering(self, input_array):
        phi, psi = self.load_filters()
        methods = ['standard', 'local', 'integral']
        if (not self.method in methods):
            raise (ValueError('method must be in {}'.format(methods)))
        if self.method == 'integral': \
                self.averaging = lambda x, j: self.backend.compute_integrals(self.backend.fft(x, inverse=True)[..., 0],
                                                                             self.integral_powers)
        elif self.method == 'local':
            self.averaging = lambda x, j: \
                self.backend._compute_local_scattering_coefs(x,
                                                             self.tensor_gaussian_filter, j, self.points)
        elif self.method == 'standard':
            self.averaging = lambda x, j: \
                self.backend._compute_standard_scattering_coefs(x,
                                                                self.tensor_gaussian_filter, self.J,
                                                                self.backend.subsample)
        L = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 0)])

        return scattering3d(input_array, pad=self.pad, unpad=self.unpad, backend=self.backend,
                            J=self.J, L=L, phi=phi, psi=psi, max_order=self.max_order)

    def forward(self, input_array):
        """
        The forward pass of 3D solid harmonic scattering
        Parameters
        ----------
        input_array: torch tensor
            input of size (batchsize, M, N, O)
        Returns
        -------
        output: tuple | torch tensor
            if max_order is 1 it returns a torch tensor with the
            first order scattering coefficients
            if max_order is 2 it returns a torch tensor with the
            first and second order scattering coefficients,
            concatenated along the feature axis
        """
        if not torch.is_tensor(input_array):
            raise (TypeError(
                'The input should be a torch.cuda.FloatTensor, '
                'a torch.FloatTensor or a torch.DoubleTensor'))

        if not input_array.is_contiguous():
            input_array = input_array.contiguous()

        if ((input_array.shape[-1] != self.P or input_array.shape[-2] != self.N
             or input_array.shape[-3] != self.M)):
            raise (RuntimeError(
                'Tensor must be of spatial size (%i,%i,%i)!' % (
                    self.M, self.N, self.P)))

        if (input_array.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]


        x = input_array.reshape((-1,) + signal_shape)
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)

        S = self.scattering(x)
        scattering_shape = S.shape[-4:]
        S = S.reshape(batch_shape + scattering_shape)
        return S


__all__ = ['HarmonicScatteringTorch3D']
