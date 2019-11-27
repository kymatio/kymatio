import torch

from .base_frontend import ScatteringBase2D
from kymatio.scattering2d.core.scattering2d import scattering2d
from ...frontend.torch_frontend import ScatteringTorch


class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        self.register_filters()

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
        """Forward pass of the scattering.

            Parameters
            ----------
            input : tensor
                Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
                arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
                :math:`n_2` is the number of input channels.

            Raises
            ------
            RuntimeError
                In the event that the input does not have at least two
                dimensions, or the tensor is not contiguous, or the tensor is
                not of the correct spatial size, padded or not.
            TypeError
                In the event that the input is not a Torch tensor.

            Returns
            -------
            S : tensor
                Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
                where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
                downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        phi, psi = self.load_filters()

        return scattering2d(input, self.pad, self.unpad, self.backend, self.J,
                            self.L, phi, psi, self.max_order)


__all__ = ['ScatteringTorch2D']
