# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

__all__ = ['Scattering3D']

import torch
from .utils import compute_integrals, subsample
 
from .backend import cdgmm3d, fft, complex_modulus, to_complex
from .filter_bank import solid_harmonic_filter_bank, gaussian_filter_bank

# TODO remove "import backend" below after implementing skcuda backend
from kymatio.scattering3d import backend


class Scattering3D(object):
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
    L: int
        number of l values

    """
    def __init__(self, J, shape, L, sigma_0):
        if backend.NAME == "skcuda":
            raise NotImplementedError(backend.skcuda_notimplementederror)
        super(Scattering3D, self).__init__()
        self.J = J
        self.shape = shape
        self.L = L
        self.sigma_0 = sigma_0

        self.build()

    def build(self):
        self.M, self.N, self.O = self.shape
        self.filters = solid_harmonic_filter_bank(
                            self.M, self.N, self.O, self.J, self.L, self.sigma_0)
        self.gaussian_filters = gaussian_filter_bank(
                                self.M, self.N, self.O, self.J + 1, self.sigma_0)

    def _fft_convolve(self, input_array, filter_array):
        """
        Computes the fourier space convolution of the input_array, 
        given in signal space, with a filter, given in fourier space.

        Parameters
        ----------

        input_array: torch tensor
            size (batchsize, M, N, O, 2)
        filter_array: torch tensor
            size (M, N, O, 2)

        Returns
        -------

        output: the result of the convolution of input_array with filter

        """
        return fft(cdgmm3d(fft(input_array, inverse=False), filter_array), inverse=True)

    def _low_pass_filter(self, input_array, j):
        """
        Computes the convolution of input_array with a lowpass filter phi_j

        Parameters
        ----------
        input_array : tensor
            size (batchsize, M, N, O, 2)

        j: int 

        Returns
        -------
        output: the result of input_array :math:`\\star phi_J`

        """
        cuda = input_array.is_cuda
        low_pass = self.gaussian_filters[j]
        if cuda:
            low_pass = low_pass.cuda()
        return self._fft_convolve(input_array, low_pass)

    def _compute_standard_scattering_coefs(self, input_array):
        """
        Computes the convolution of input_array with a lowpass filter phi_J 
        and downsamples by a factor J.

        Parameters
        ----------
        input_array: torch tensor of size (batchsize, M, N, O, 2)

        Returns
        -------
        output: the result of input_array \\star phi_J downsampled by a factor J

        """
        convolved_input = self._low_pass_filter(input_array, self.J)
        return subsample(convolved_input, self.J).view(
            input_array.size(0), -1, 1)

    def _compute_local_scattering_coefs(self, input_array, points, j):
        """
        Computes the convolution of input_array with a lowpass filter phi_j and 
        and returns the value of the output at particular points

        Parameters
        ----------
        input_array: torch tensor
            size (batchsize, M, N, O, 2)
        points: torch tensor
            size (batchsize, number of points, 3)
        j: int
            the lowpass scale j of phi_j

        Returns
        -------
        output: torch tensor of size (batchsize, number of points, 1) with
                the values of the lowpass filtered moduli at the points given.

        """
        local_coefs = torch.zeros(input_array.size(0), points.size(1), 1)
        convolved_input = self._low_pass_filter(input_array, j+1)
        for i in range(input_array.size(0)):
            for j in range(points[i].size(0)):
                x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
                local_coefs[i, j, 0] = convolved_input[
                                                  i, int(x), int(y), int(z), 0]
        return local_coefs

    def _compute_scattering_coefs(self, input_array, method, args, j):
        """
        Computes the scattering coefficients out with any of the three methods 
        'standard', 'local', 'integral'

        Parameters
        ----------
        input_array : torch tensor
            size (batchsize, M, N, O, 2)
        method : string
            method name with three possible values ("standard", "local", "integral")
        args : dict
            method specific arguments. It methods is equal to "standard", then one
            expects the array args['integral_powers'] to be a list that holds
            the exponents the moduli. It should be raised to before calculating
            the integrals. If method is equal to "local", args['points'] must contain
            a torch tensor of size (batchsize, number of points, 3) the points in
            coordinate space at which you want the moduli sampled
        j : int
            lowpass scale j of :math:`\\phi_j`

        Returns
        -------
        output: torch tensor 
            The scattering coefficients as given by different methods.

        """
        methods = ['standard', 'local', 'integral']
        if (not method in methods):
            raise(ValueError('method must be in {}'.format(methods)))
        if method == 'integral':
            return compute_integrals(input_array[..., 0],
                                     args['integral_powers'])
        elif method == 'local':
            return self._compute_local_scattering_coefs(input_array,
                                                        args['points'], j)
        elif method == 'standard':
            return self._compute_standard_scattering_coefs(input_array)

    def _rotation_covariant_convolution_and_modulus(self, input_array, l, j):
        """
        Computes the convolution with a set of solid harmonics of scale j and 
        degree l and returns the square root of their squared sum over m

        Parameters
        ----------
        input_array : tensor
            size (batchsize, M, N, O, 2)
        l : int
            solid harmonic degree l

        j : int
            solid harmonic scale j

        Returns
        -------

        output : torch tensor
            tensor of the same size as input_array. It holds the output of
            the operation::

            $\sqrt{\sum_m (\text{input}_\text{array} \star \psi_{j,l,m})^2)}

            which is covariant to 3D translations and rotations

        """
        cuda = input_array.is_cuda
        filters_l_j = self.filters[l][j]
        if cuda:
            filters_l_j = filters_l_j.cuda()
        convolution_modulus = input_array.new(input_array.size()).fill_(0)
        for m in range(filters_l_j.size(0)):
            convolution_modulus[..., 0] += (self._fft_convolve(
                input_array, filters_l_j[m]) ** 2).sum(-1)
        return torch.sqrt(convolution_modulus)

    def _convolution_and_modulus(self, input_array, l, j, m=0):
        """
        Computes the convolution with a set of solid harmonics of scale j and 
        degree l and returns the square root of their squared sum over m

        Parameters
        ----------
        input_array: torch tensor
            size (batchsize, M, N, O, 2)
        l : int
            solid harmonic degree l
        j : int
            solid harmonic scale j
        m : int, optional
            solid harmonic rank m (defaults to 0)

        Returns
        -------
        output: torch tensor 
                tensor of the same size as input_array. It holds the output of the
                operation::

                .. math:: \\text{input}_\\text{array} \\star \\psi_{j,l,m})

        """
        cuda = input_array.is_cuda
        filters_l_m_j = self.filters[l][j][m]
        if cuda:
            filters_l_m_j = filters_l_m_j.cuda()
        return complex_modulus(self._fft_convolve(input_array, filters_l_m_j))

    def _check_input(self, input_array):
        if not torch.is_tensor(input_array):
            raise(TypeError(
                'The input should be a torch.cuda.FloatTensor, '
                'a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input_array.is_contiguous()):
            input_array = input_array.contiguous()

        if((input_array.size(-1) != self.O or input_array.size(-2) != self.N 
                                           or input_array.size(-3) != self.M)):
            raise (RuntimeError(
                'Tensor must be of spatial size (%i,%i,%i)!' % (
                                                      self.M, self.N, self.O)))

        if (input_array.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

    def forward(self, input_array, order_2=True, rotation_covariant=True,
                method='standard', points=None, integral_powers=(.5, 1., 2.)):
        """
        The forward pass of 3D solid harmonic scattering

        Parameters
        ----------
        input_array: torch tensor 
            input of size (batchsize, M, N, O)
        order_2: bool, optional
            if set to False|True it also excludes|includes second order
            scattering coefficients (default: True).
        rotation_covariant: bool, optional
            if set to True the first order moduli take the form::

            $\sqrt(\sum_m (input_array \star \psi_{j,l,m})^2))$

            if set to False the first order moduli take the form::

            $input_array \star \psi_{j,l,m})$

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

        Returns
        -------
        output: tuple | torch tensor
            if order_2 is false it returns a torch tensor with the
            first order scattering coefficients
            if order_2 is true it returns a tuple with two elements,
            the first and second order scattering coefficients

        """
        self._check_input(input_array)
        if rotation_covariant:
            convolution_and_modulus = (
                self._rotation_covariant_convolution_and_modulus)
        else:
            convolution_and_modulus = self._convolution_and_modulus

        compute_scattering_coefs = self._compute_scattering_coefs

        s_order_1 = []
        s_order_2 = []
        _input = to_complex(input_array)

        method_args = dict(points=points, integral_powers=integral_powers)

        for l in range(self.L+1):
            s_order_1_l, s_order_2_l = [], []
            for j_1 in range(self.J+1):
                conv_modulus = convolution_and_modulus(_input, l, j_1)
                s_order_1_l.append(compute_scattering_coefs(
                    conv_modulus, method, method_args, j_1))
                if not order_2:
                    continue
                for j_2 in range(j_1+1, self.J+1):
                    conv_modulus_2 = convolution_and_modulus(
                        conv_modulus, l, j_2)
                    s_order_2_l.append(compute_scattering_coefs(
                        conv_modulus_2, method, method_args, j_2))
            s_order_1.append(torch.cat(s_order_1_l, -1))
            if order_2:
                s_order_2.append(torch.cat(s_order_2_l, -1))

        if order_2:
            return torch.cat(
                [torch.stack(s_order_1, dim=-1),
                torch.stack(s_order_2, dim=-1)], -2)
        else:
            return torch.stack(s_order_1, dim=-1)


    __call__ = forward

