# Authors: Mathieu Andreux, Joakim Anden, Edouard Oyallon
# Scientific Ancestry: Joakim Anden, Mathieu Andreux, Vincent Lostanlen

import math
import numbers

import torch

from ...frontend.torch_frontend import ScatteringTorch
from .general_frontend import Scattering1DGeneral

from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.utils import precompute_size_scattering

__all__ = ['Scattering1DTorch']

class Scattering1DTorch(ScatteringTorch, Scattering1DGeneral):
    def __init__(self, J, shape, Q=1, max_order=2, average=True,
                 oversampling=0, vectorize=True, backend=None):
        ScatteringTorch.__init__(self)
        
        # Store the parameters
        # Build internal values
        Scattering1DGeneral.__init__(self, J, shape, Q, max_order, average, oversampling, vectorize, backend)

        self.backend_setup()    
        self.register_filters()
        
    def backend_setup(self):
        """Set up backend."""
        if not self.backend:
            from ..backend.torch_backend import backend
            self.backend = backend
        elif not self.backend.name.startswith('torch'):
            raise RuntimeError('This backend is not supported.')

    def register_filters(self):
        """This function saves filters as module's buffers."""
        n = 0
        # prepare for pytorch
        for k in self.phi_f.keys():
            n = self.save_key(k, self.phi_f, n)
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                n = self.save_key(sub_k, psi_f, n)
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                n = self.save_key(sub_k, psi_f, n)

    def save_key(self, k, filters, n):
        if type(k) != str:
            #view(-1, 1).repeat(1, 2) because real numbers!
            filters[k] = torch.from_numpy(filters[k]).float().view(-1, 1).repeat(1, 2)
            self.register_buffer('tensor' + str(n), filters[k])
            n += 1
        return n

    def scattering(self, x):
        """Apply the scattering transform.

        Given an input Tensor of size `(B, T0)`, where `B` is the batch
        size and `T0` is the length of the individual signals, this function
        computes its scattering transform. If the `vectorize` flag is set to
        `True`, the output is in the form of a Tensor or size `(B, C, T1)`,
        where `T1` is the signal length after subsampling to the scale `2**J`
        (with the appropriate oversampling factor to reduce aliasing), and
        `C` is the number of scattering coefficients.  If `vectorize` is set
        `False`, however, the output is a dictionary containing `C` keys, each
        a tuple whose length corresponds to the scattering order and whose
        elements are the sequence of filter indices used.

        Furthermore, if the `average` flag is set to `False`, these outputs
        are not averaged, but are simply the wavelet modulus coefficients of
        the filters.

        Parameters
        ----------
        x : tensor
            An input Tensor of size `(B, T0)`.

        Returns
        -------
        S : tensor or dictionary
            If the `vectorize` flag is `True`, the output is a Tensor
            containing the scattering coefficients, while if `vectorize`
            is `False`, it is a dictionary indexed by tuples of filter indices.

        """
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            if not(self.average):
                raise ValueError(
                    'Options average=False and vectorize=True are ' +
                    'mutually incompatible. Please set vectorize to False.')
            size_scattering = precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        self.retrive_filters()
        
        S = scattering1d(x, self.backend, self.J, self.psi1_f, self.psi2_f, self.phi_f,\
                         max_order=self.max_order, average=self.average,
                       pad_left=self.pad_left, pad_right=self.pad_right,
                       ind_start=self.ind_start, ind_end=self.ind_end,
                       oversampling=self.oversampling,
                       vectorize=self.vectorize,
                       size_scattering=size_scattering)

        if self.vectorize:
            scattering_shape = S.shape[-2:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            for k, v in S.items():
                scattering_shape = v.shape[-2:]
                S[k] = v.reshape(batch_shape + scattering_shape)

        return S

    def retrive_filters(self):
        n = 0
        buffer_dict = dict(self.named_buffers())
        
        for k in self.phi_f.keys():
            n = self.get_buffer(k, self.phi_f, n, buffer_dict)
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                n = self.get_buffer(sub_k, psi_f, n, buffer_dict)
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                n = self.get_buffer(sub_k, psi_f, n, buffer_dict)

    def get_buffer(self, k, filters, n, buffer_dict):
        if type(k) != str:
            filters[k] = buffer_dict['tensor' + str(n)]
            n += 1
        return n

    def loginfo(self):
        return 'Torch frontend is used.'

