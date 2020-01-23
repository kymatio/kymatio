# Authors: Mathieu Andreux, Joakim Anden, Edouard Oyallon
# Scientific Ancestry: Joakim Anden, Mathieu Andreux, Vincent Lostanlen

from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering1d import scattering1d
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, max_order=2, average=True, oversampling=0, vectorize=True, backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average, oversampling, vectorize, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)

    def scattering(self, x):
        """Apply the scattering transform

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
            if not (self.average):
                raise ValueError(
                    'Options average=False and vectorize=True are ' +
                    'mutually incompatible. Please set vectorize to False.')
            size_scattering = precompute_size_scattering(
                self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0

        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.psi1_f, self.psi2_f,
                         self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left,
                         pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling, vectorize=self.vectorize, size_scattering=size_scattering)

        if self.vectorize:
            scattering_shape = S.shape[-2:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            for k, v in S.items():
                scattering_shape = v.shape[-2:]
                S[k] = v.reshape(batch_shape + scattering_shape)

        return S


__all__ = ['ScatteringNumPy1D']
