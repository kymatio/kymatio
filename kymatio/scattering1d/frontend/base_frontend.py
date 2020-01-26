from ...frontend.base_frontend import ScatteringBase
import math
import numbers

import numpy as np

from ..filter_bank import scattering_filter_factory
from ..utils import (compute_border_indices, compute_padding, compute_minimum_support_to_pad,
compute_meta_scattering, precompute_size_scattering)


class ScatteringBase1D(ScatteringBase):
    """The 1D scattering transform
    
        The scattering transform computes a cascade of wavelet transforms
        alternated with a complex modulus non-linearity. The scattering transform
        of a 1D signal :math:`x(t)` may be written as
    
            $S_J x = [S_J^{(0)} x, S_J^{(1)} x, S_J^{(2)} x]$
    
        where
    
            $S_J^{(0)} x(t) = x \\star \\phi_J(t)$,
    
            $S_J^{(1)} x(t, \\lambda) =|x \\star \\psi_\\lambda^{(1)}| \\star \\phi_J$, and
    
            $S_J^{(2)} x(t, \\lambda, \\mu) = |\\,| x \\star \\psi_\\lambda^{(1)}| \\star \\psi_\\mu^{(2)} | \\star \\phi_J$.
    
        In the above formulas, :math:`\\star` denotes convolution in time. The
        filters $\\psi_\\lambda^{(1)}(t)$ and $\\psi_\\mu^{(2)}(t)$
        are analytic wavelets with center frequencies $\\lambda$ and
        $\\mu$, while $\\phi_J(t)$ is a real lowpass filter centered
        at the zero frequency.
    
        The `Scattering1D` class implements the 1D scattering transform for a
        given set of filters whose parameters are specified at initialization.
        While the wavelets are fixed, other parameters may be changed after the
        object is created, such as whether to compute all of :math:`S_J^{(0)} x`,
        $S_J^{(1)} x$, and $S_J^{(2)} x$ or just $S_J^{(0)} x$
        and $S_J^{(1)} x$.
    
        The scattering transform may be computed on the CPU (the default) or a
        GPU, if available. A `Scattering1D` object may be transferred from one
        to the other using the `cuda()` and `cpu()` methods.
    
        Given an input Tensor `x` of size `(B, T)`, where `B` is the number of
        signals to transform (the batch size) and `T` is the length of the signal,
        we compute its scattering transform by passing it to the `forward()`
        method.
    
        Example
        -------
        ::
    
            # Set the parameters of the scattering transform.
            J = 6
            T = 2**13
            Q = 8
    
            # Generate a sample signal.
            x = torch.randn(1, 1, T)
    
            # Define a Scattering1D object.
            S = Scattering1D(J, T, Q)
    
            # Calculate the scattering transform.
            Sx = S.forward(x)
    
        Above, the length of the signal is `T = 2**13 = 8192`, while the maximum
        scale of the scattering transform is set to `2**J = 2**6 = 64`. The
        time-frequency resolution of the first-order wavelets
        :math:`\\psi_\\lambda^{(1)}(t)` is set to `Q = 8` wavelets per octave.
        The second-order wavelets :math:`\\psi_\\mu^{(2)}(t)` always have one
        wavelet per octave.
    
        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by `2**J`.
        T : int
            The length of the input signals.
        Q : int >= 1
            The number of first-order wavelets per octave (second-order wavelets
            are fixed to one wavelet per octave). Defaults to `1`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be either
            `1` or `2`. Defaults to `2`.
        average : boolean, optional
            Determines whether the output is averaged in time or not. The averaged
            output corresponds to the standard scattering transform, while the
            un-averaged output skips the last convolution by :math:`\\phi_J(t)`.
            This parameter may be modified after object creation.
            Defaults to `True`.
        oversampling : integer >= 0, optional
            Controls the oversampling factor relative to the default as a power
            of two. Since the convolving by wavelets (or lowpass filters) and
            taking the modulus reduces the high-frequency content of the signal,
            we can subsample to save space and improve performance. However, this
            may reduce precision in the calculation. If this is not desirable,
            `oversampling` can be set to a large value to prevent too much
            subsampling. This parameter may be modified after object creation.
            Defaults to `0`.
        vectorize : boolean, optional
            Determines wheter to return a vectorized scattering transform (that
            is, a large array containing the output) or a dictionary (where each
            entry corresponds to a separate scattering coefficient). This parameter
            may be modified after object creation. Defaults to True.
    
        Attributes
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by `2**J`.
        shape : int
            The length of the input signals.
        Q : int
            The number of first-order wavelets per octave (second-order wavelets
            are fixed to one wavelet per octave).
        J_pad : int
            The logarithm of the padded length of the signals.
        pad_left : int
            The amount of padding to the left of the signal.
        pad_right : int
            The amount of padding to the right of the signal.
        phi_f : dictionary
            A dictionary containing the lowpass filter at all resolutions. See
            `filter_bank.scattering_filter_factory` for an exact description.
        psi1_f : dictionary
            A dictionary containing all the first-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an exact
            description.
        psi2_f : dictionary
            A dictionary containing all the second-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an exact
            description.
            description
        max_order : int
            The maximum scattering order of the transform.
        average : boolean
            Controls whether the output should be averaged (the standard
            scattering transform) or not (resulting in wavelet modulus
            coefficients). Note that to obtain unaveraged output, the `vectorize`
            flag must be set to `False`.
        oversampling : int
            The number of powers of two to oversample the output compared to the
            default subsampling rate determined from the filters.
        vectorize : boolean
            Controls whether the output should be vectorized into a single Tensor
            or collected into a dictionary. For more details, see the
            documentation for `forward()`.
        """
    def __init__(self, J, shape, Q=1, max_order=2, average=True, oversampling=0, vectorize=True, backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.vectorize = vectorize
        self.backend = backend

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """

        # check the shape
        if isinstance(self.shape, numbers.Integral):
            self.T = self.shape
        elif isinstance(self.shape, tuple):
            self.T = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")

        # Compute the minimum support to pad (ideally)
        min_to_pad = compute_minimum_support_to_pad(
            self.T, self.J, self.Q)
        # to avoid padding more than T - 1 on the left and on the right,
        # since otherwise torch sends nans
        J_max_support = int(np.floor(np.log2(3 * self.T - 2)))
        self.J_pad = min(int(np.ceil(np.log2(self.T + 2 * min_to_pad))),
                         J_max_support)
        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.T)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.J, self.pad_left, self.pad_left + self.T)

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f, _ = scattering_filter_factory(
            self.J_pad, self.J, self.Q)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(self.J, self.Q, max_order=self.max_order)

    def output_size(self, detail=False):
        """Get size of the scattering transform

        Calls the static method `precompute_size_scattering()` with the
        parameters of the transform object.

        Parameters
        ----------
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        ------
        size : int or tuple
            See the documentation for `precompute_size_scattering()`.
        """

        return precompute_size_scattering(
            self.J, self.Q, max_order=self.max_order, detail=detail)


__all__ = ['ScatteringBase1D']
