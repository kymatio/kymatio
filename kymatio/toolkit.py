# -*- coding: utf-8 -*-
"""Convenience utilities."""
import numpy as np
import scipy.signal
import warnings
from itertools import zip_longest, chain
from copy import deepcopy


def drop_batch_dim_jtfs(Scx, sample_idx=0):
    """Index into dim0 with `sample_idx` for every JTFS coefficient, and
    drop that dimension.

    Doesn't modify input:
        - dict/list: new list/dict (with copied meta if applicable)
        - array: new object but shared storage with original array (so original
          variable reference points to unindexed array).
    """
    fn = lambda x: x[sample_idx]
    return _iterate_apply(Scx, fn)


def jtfs_to_numpy(Scx):
    """Convert PyTorch/TensorFlow tensors to numpy arrays, with meta copied,
    and without affecting original data structures.
    """
    B = ExtendedUnifiedBackend(Scx)
    return _iterate_apply(Scx, B.numpy)


def _iterate_apply(Scx, fn):
    def get_meta(s):
        return {k: v for k, v in s.items() if not hasattr(v, 'ndim')}

    if isinstance(Scx, dict):
        out = {}  # don't modify source dict
        for pair in Scx:
            if isinstance(Scx[pair], list):
                out[pair] = []
                for i, s in enumerate(Scx[pair]):
                    out[pair].append(get_meta(s))
                    out[pair][i]['coef'] = fn(s['coef'])
            else:
                out[pair] = fn(Scx[pair])
    elif isinstance(Scx, list):
        out = []  # don't modify source list
        for s in Scx:
            o = get_meta(s)
            o['coef'] = fn(s['coef'])
            out.append(o)
    elif isinstance(Scx, tuple):  # out_type=='array' && out_3D==True
        out = (fn(Scx[0]), fn(Scx[1]))
    elif hasattr(Scx, 'ndim'):
        out = fn(Scx)
    else:
        raise ValueError(("unrecognized input type: {}; must be as returned by "
                          "`jtfs(x)`.").format(type(Scx)))
    return out


def normalize(X, rscaling='l1', mean_axis=(1, 2), std_axis=(1, 2), C=None,
              mu=None, C_mult=None):
    """Log-normalize + (optionally) standardize coefficients for learning
    algorithm suitability.

    Is a modification of Eq. 10 of https://arxiv.org/pdf/2007.10926.pdf
    For exact match (minus temporal global averaging), set `rscaling=None`
    and `mean_axis=std_axis=(0, 2)`.

    Parameters
    ----------
    X : tensor
        Nonnegative tensor with dimensions `(samples, features, spatial)`.
        If there's more than one `features` or `spatial` dimensions, flatten
        before passing.
        (Obtain tensor via e.g. `pack_coeffs_jtfs(Scx)`, or `out_type='array'`.)

    rscaling : str / None  # TODO deprecate
        If not None, string one of: 'l1', 'l2'. Interacts with `std_axis`
        and `mean_axis`. See "Relative scaling" below.
        Must be None if spatial dim is `1`.

    std_axis : tuple[int] / int / None
        If not None, will unit-variance after `rscaling` along specified axes.

    mean_axis : tuple[int] / int / None
        If not None, will zero-mean before `rscaling` along specified axes.

    C : float / None
        `log(1 + X * C / median)`.
        Greater will bring more disparate values closer. Too great will equalize
        too much, too low will have minimal effect.

        Defaults to `5 / mean(abs(X / mu))`, which should yield moderate
        contraction for a variety of signals. This was computed on a mixture
        of random processes, with outliers, and may not generalize to all signals.

    mu : float / None
        In case precomputed; see "Online computation".

    C_mult : float / None
        Multiplies `C`. Useful if the default `C` compute scheme is appropriate
        but needs adjusting. Defaults to `5` if `C` is None, else to `1`.

    Returns
    -------
    Xnorm : tensor
        Normalized `X`.

    transformers : type / None
        Transformer objects, if applicable - else None.

    Relative scaling
    ----------------

    Scaling `features` independently changes the relative norms bewteen them.

      - If a signal rarely has high frequencies and low are dominant, for example,
        then post-normalization this nuance is lost and highs and lows are brought
        to a common norm - which may be undesired.
      - SNR is lowered, as low signal contents that are dominated by noise
        or float inaccuracies are amplified.
      - Convolutions over `features` dims are invalidated (as it's akin to
        standardizing individual time steps in 1D convolution); e.g. if
        normalizing on per-`n1` basis, then we can no longer do 2D convs
        over the joint `(n1, time)` pairs.
      - To keep convs valid, all spatial dims that are convolved over must be
        standardized by the same factor - i.e. same `mean` and `std`. `rscaling`
        also accounts for rescaling due to log.

    Regardless, this "channel normalization" has been used with success in
    variuous settings; above are but points worth noting.

    Online computation
    ------------------

    Any computation with `axis` that includes `0` requires simultaneous access
    to all samples. This poses a problem in two settings:

        1. Insufficient RAM. The solution is to write an *equivalent* computation
           that aggregates statistics one sample at a time. E.g. for `mu`:

               Xsum = []
               for x in dataset:
                   Xsum.append(B.sum(x, axis=-1, keepdims=True))
               mu = B.median(B.vstack(Xsum), axis=0, keepdims=True)

        2. Streaming / new samples. In this case we must reuse parameters computed
           over e.g. entire train set.

    Computations over all axes *except* `0` are done on per-sample basis, which
    means not having to rely on other samples - but also an inability to do so
    (i.e. to precompute and reuse params).
    """
    # validate args & set defaults ###########################################
    supported = ('l1', 'l2', None)
    if rscaling not in supported:
        raise ValueError(("unsupported `rscaling` {}; must be one of: {}"
                          ).format(rscaling, ', '.join(supported)))
    if X.ndim != 3:
        raise ValueError("input must be 3D, `(samples, features, spatial)` - "
                         "got %s" % str(X.shape))
    B = ExtendedUnifiedBackend(X)

    # check input values
    if B.min(X) < 0:
        warnings.warn("`X` must be non-negative; will take modulus.")
        X = B.abs(X)
    # convert axes to positive
    axes = [mean_axis, std_axis]
    for i, ax in enumerate(axes):
        if ax is None:
            continue
        ax = ax if isinstance(ax, (list, tuple)) else [ax]
        ax = list(ax)
        for j, a in enumerate(ax):
            if a < 0:
                ax[j] = X.ndim + a
        axes[i] = tuple(ax)
    mean_axis, std_axis = axes

    # check input dims
    dim_ones = tuple(d for d in range(X.ndim) if X.shape[d] == 1)
    if dim_ones != ():
        def check_dims(g, name):
            g = g if isinstance(g, (tuple, list)) else (g,)
            if all(dim in dim_ones for dim in g):
                raise ValueError("input dims cannot be `1` along same dims as "
                                 "`{}` (gives NaNs); got X.shape == {}, "
                                 "{} = {}".format(name, X.shape, name, mean_axis))

        check_dims(mean_axis, 'mean_axis')
        check_dims(std_axis,  'std_axis')
        # check mu
        if mu is None and 0 in dim_ones and 2 in dim_ones:
            raise ValueError("input dims cannot be `1` along dims 0 and 2 "
                             "if `mu` is None (gives NaNs); "
                             "got X.shape == {}".format(X.shape))
        # check rscaling
        if rscaling is not None and 2 in dim_ones:
            raise ValueError("input dims cannot be `1` along dim 2 "
                             "if `rscaling` is not None (gives NaNs); "
                             "got X.shape == {}".format(X.shape))

    # main transform #########################################################
    if mu is None:
        # spatial sum (integral)
        Xsum = B.sum(X, axis=-1, keepdims=True)
        # sample median
        mu = B.median(Xsum, axis=0, keepdims=True)

    def sparse_mean(x, div=100, iters=4):
        """Mean of non-negligible points"""
        m = x.mean()
        for _ in range(iters - 1):
            m = x[x > m / div].mean()
        return m

    # rescale
    Xnorm = X / mu
    # contraction factor
    if C_mult is None:
        C_mult = 5 if C is None else 1
    if C is None:
        C = 1 / sparse_mean(B.abs(Xnorm), iters=4)
    C *= C_mult
    # log
    Xnorm = B.log(1 + Xnorm * C)

    # standardization + relative scaling #####################################
    if mean_axis is not None:
        Xnorm -= B.mean(Xnorm, axis=mean_axis, keepdims=True)

    if rscaling is not None:
        # rescale per `feature` and per `sample`
        # so compute statistics along `spatial`
        kw = dict(axis=-1, keepdims=True)
        ord = (2 if rscaling == 'l2' else 1)
        # produces zeros if spatial dim is of size 1 or values are equal (dc)
        Xstd = X - B.mean(X, **kw)
        norms_orig = B.norm(Xstd,  ord=ord, **kw)
        norms_now  = B.norm(Xnorm, ord=ord, **kw)
        norms = norms_orig / norms_now
        if np.any(np.isnan(norms)) or np.any(np.isinf(norms)):
            raise Exception
        if norms.mean() == 0:
            raise Exception
        Xnorm *= (norms / norms.mean())

    if std_axis is not None:
        Xnorm /= B.std(Xnorm, axis=std_axis, keepdims=True)

    return Xnorm


def pack_coeffs_jtfs(Scx, meta, structure=1, sample_idx=None,
                     separate_lowpass=False, sampling_psi_fr=None, out_3D=None,
                     debug=False, recursive=False):
    """Packs efficiently JTFS coefficients into one of valid 4D structures.

    Parameters
    ----------
    Scx : tensor/list/dict
        JTFS output. Must have `out_type` 'dict:array' or 'dict:list'
        and `average=True`.

    meta : dict
        JTFS meta.

    structure : int / None
        Structure to pack `Scx` into (see "Structures" below), integer 1 to 4.
        Will pack into a structure even if not suitable for convolution (as
        determined by JTFS parameters); see "Structures" if convs are relevant.

          - If can pack into one structure, can pack into any other (1 to 4).
          - 5 to 8 aren't implemented since they're what's already returned
            as output.

    sample_idx : int / None
        Index of sample in batched input to pack. If None (default), will
        pack all samples.
        Returns 5D if `not None` *and* there's more than one sample.

    separate_lowpass : bool (default False)
        If True, will pack spinned (`psi_t * psi_f_up`, `psi_t * psi_f_down`)
        and lowpass (`phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`) pairs
        separately. Recommended for convolutions (see Structures & Ordinality).

    sampling_psi_fr : str / None
        Used for sanity check for padding along `n1_fr`.
        Must match what was passed to `TimeFrequencyScattering1D`.
        If None, will assume library default.

    out_3D : bool / None
        Used for sanity check for padding along `n1`
        (enforces same `n1` per `n2`).

    debug : bool (defualt False)
        If True, coefficient values will be replaced by meta `n` values for
        debugging purposes, where the last dim is size 4 and contains
        `(n1_fr, n2, n1, time)` assuming `structure == 1`.

    Returns
    -------
    out: tensor / tuple[tensor]
        Packed `Scx`, depending on `structure` and `separate_lowpass`:

          - 1: `out` if False else
               `(out, out_phi_f, out_phi_t)`
          - 2: same as 1
          - 3: `(out_up, out_down, out_phi_f)` if False else
               `(out_up, out_down, out_phi_f, out_phi_t)`
          - 4: `(out_up, out_down)` if False else
               `(out_up, out_down, out_phi_t)`

        `out_phi_t` is `phi_t * psi_f` and `phi_t * phi_f` concatenated.
        `out_phi_f` is `psi_t * phi_f` for all configs except
        `3, True`, where it is concatenated with `phi_t * phi_f`.

        For further info, see "Structures", "Parameter effects", and "Notes".

    Structures
    ----------
    Assuming `aligned=True`, then for `average, average_fr`, the following form
    valid convolution structures:

      1. `True, True*`:  3D/4D*, `(n1_fr, n2, n1, time)`
      2. `True, True*`:  2D/4D*, `(n2, n1_fr, n1, time)`
      3. `True, True*`:  4D,     `(n2, n1_fr//2,     n1, time)`*2,
                                 `(n2, 1, n1, time)`
      4. `True, True*`:  2D/4D*, `(n2, n1_fr//2 + 1, n1, time)`*2
      5. `True, True*`:  2D/3D*, `(n2 * n1_fr, n1, time)`
      6. `True, False`:  1D/2D*, `(n2 * n1_fr * n1, time)`
      7. `False, True`:  list of variable length 1D tensors
      8. `False, False`: list of variable length 1D tensors

    where

      - n1: frequency [Hz], first-order temporal variation
      - n2: frequency [Hz], second-order temporal variation
        (frequency of amplitude modulation)
      - n1_fr: quefrency [cycles/octave], first-order frequential variation
        (frequency of frequency modulation bands, roughly. More precisely,
         correlates with frequential bands (independent components/modes) of
         varying widths, decay factors, and recurrences, per temporal slice)
      - time: time [sec]
      - The actual units are dimensionless, "Hz" and "sec" are an example.
        To convert, multiply by sampling rate `fs`.
      - For convolutions, first dim is assumed to be channels (unless doing
        4D convs).
      - `True*` indicates a "soft requirement"; as long as `aligned=True`,
        `False` can be fully compensated with padding.
        Since 5 isn't implemented with `False`, it can be obtained from `False`
        by reshaping one of 1-4.
      - `2D/4D*` means 3D/4D convolutions aren't strictly valid for convolving
        over trailing (last) dimensions (see below), but 1D/2D are.
        `3D` means 1D, 2D, 3D are all valid.
      - Structure 3 is 3D/4D-valid only if one deems valid the disjoint
        representation with separate convs over spinned and lowpassed
        (thus convs over lowpassed-only coeffs are deemed valid) - or if one
        opts to exclude the lowpassed pairs.
      - Structure 4 is 3D/4D-valid only if one deems valid convolving over both
        lowpassed and spinned coefficients.

    The interpretations for convolution (and equivalently, spatial coherence)
    are as follows:

        1. The true JTFS structure. `(n2, n1, time)` are ordinal and thus
           valid dimensions for 3D convolution (if all `phi` pairs are excluded,
           which isn't default behavior; see "Ordinality").
        2. It's a dim-permuted 1, but last three dimensions are no longer ordinal
           and don't necessarily form a valid convolution pair.
           This is the preferred structure for conceptualizing or debugging as
           it's how the computation graph unfolds (and so does information
           density, as `N_fr` varies along `n2`).
        3. It's 2, but split into ordinal pairs - `out_up, out_down, out_phi`
           suited for convolving over last three dims. These still include
           `phi_t * psi_f` pairs, so for strict ordinality the first slice
           should drop (e.g. `out_up[1:]`).
        4. It's 3, but only `out_up, out_down`, and each includes `psi_t * phi_f`.
           If this "soft ordinality" is acceptable then `phi_t * psi_f` pairs
           should be kept.
        5. `n2` and `n1_fr` are flattened into one dimension. The resulting
           3D structure is suitable for 2D convolutions along `(n1, time)`.
        6. `n2`, `n1_fr`, and `n1` are flattened into one dimension. The resulting
           2D structure is suitable for 1D convolutions along `time`.
        7. `time` is variable; structue not suitable for convolution.
        8. `time` and `n1` is variable; structure not suitable for convolution.

    Structures not suited for convolutions may be suited for other transforms,
    e.g. Dense or Graph Neural Networks.

    Helpful visuals:
      https://github.com/kymatio/kymatio/discussions/708#discussioncomment-1624521

    Ordinality
    ----------
    Coefficients are "ordinal" if their generating wavelets are spaced linearly
    in log space. The lowpass filter is equivalently an infinite scale wavelet,
    thus it breaks ordinality. Opposite spins require stepping over the lowpass
    and are hence disqualified.

    Above is strictly true in continuous time. In a discrete setting, however,
    the largest possible non-dc scale is far from infinite. A 2D lowpass wavelet
    is somewhat interpretable as a subsequent scaling and rotation of the largest
    scale bandpass, as the bandpass itself is such a scaling and rotation of its
    preceding bandpass.

    Nonetheless, a lowpass is an averaging rather than modulation extracting
    filter: its physical units differ, and it has zero FDTS sensitivity - and
    this is a stronger objection for convolution. Further, when convolving over
    modulus of wavelet transform (as frequential scattering does), the dc bin
    is most often dominant, and by a lot - thus without proper renormalization
    it will drown out the bandpass coefficients in concatenation.

    The safest configuration for convolution thus excludes all lowpass pairs:
    `phi_t * phi_f`, `phi_t * psi_f`, and `psi_t * phi_f`; these can be convolved
    over separately.

    Parameter effects
    -----------------
    `average` and `average_fr` are described in "Structures". Additionally:

      - aligned:
        - True: enables the true JTFS structure (every structure in 1-6 is
          as described).
        - False: yields variable stride along `n1`, disqualifying it from
          3D convs along `(n2, n1, time)`. However, assuming semi-ordinality
          is acceptable, then each `n2` slice in `(n2, n1_fr, n1, time)`, i.e.
          `(n1_fr, n1, time)`, has the same stride, and forms valid conv pair
          (so use 3 or 4). Other structures require similar accounting.
          Rules out structure 1 for 3D/4D convs.
      - out_3D:
        - True: enforces same freq conv stride on *per-`n2`* basis, enabling
          3D convs even if `aligned=False`.
      - sampling_psi_fr:
        - 'resample': enables the true JTFS structure.
        - 'exclude': enables the true JTFS structure (it's simply a subset of
          'resample'). However, this involves large amounts of zero-padding to
          fill the missing convolutions and enable 4D concatenation.
        - 'recalibrate': breaks the true JTFS structure. `n1_fr` frequencies
          and widths now vary with `n2`, which isn't spatially coherent in 4D.
          It also renders `aligned=True` a pseudo-alignment.
          Like with `aligned=False`, alignment and coherence is preserved on
          per-`n2` basis, retaining the true structure in a piecewise manner.
          Rules out structure 1 for 3D/4D convs.
      - average:
        - It's possible to support `False` the same way `average_fr=False` is
          supported, but this isn't implemented.

    Notes
    -----
      - Coefficients will be sorted in order of increasing center frequency -
        along `n2`, and `n1_fr` for spin down (which makes it so that both spins
        are increasing away from `psi_t * phi_f`). The overall packing is
        structured same as in `kymatio.visuals.filterbank_jtfs_2d()`.

      - Method assumes `out_exclude=None` if `not separate_lowpass` - else,
        the following are allowed to be excluded: 'phi_t * psi_f',
        'phi_t * phi_f', and if `structure != 4`, 'psi_t * phi_f'.

      - The built-in energy renormalization includes doubling the energy
        of `phi_t * psi_f` pairs to compensate for computing only once (for
        just one spin since it's identical to other spin), while here it's
        also packed twice; to compensate, its energy is halved before packing.

      - Energy duplication isn't avoided for all `structure` & `separate_lowpass`:
          - `3, separate_lowpass`: packs the `phi_t * phi_f` pair twice -
            with `phi_t * psi_f`, and with `psi_t * phi_f`.
            `out_phi_f` always concats with `phi_t * phi_f` for `3` since
            `phi_f` is never concat with spinned, so it can't concat with
            `phi_t` pairs as usual.
          - `4, not separate_lowpass`: packs `phi_t * phi_f` and `psi_t * phi_f`
            pairs twice, once for each spin.
          - `4, separate_lowpass`: packs `psi_t * phi_f` pairs twice, once for
            each spin.
          - Note both `3` and `4` pack `phi_t * psi_f` pairs twice if
            `not separate_lowpass`, but the energy is halved anyway and hence
            not duped.
    """
    B = ExtendedUnifiedBackend(Scx)

    def combined_to_tensor(combined_all, recursive):
        def process_dims(o):
            if recursive:
                assert o.ndim == 5, o.shape
            else:
                assert o.ndim == 4, o.shape
                o = o[None]
            return o

        def not_none(x):
            return (x is not None if not recursive else
                    all(_x is not None for _x in x))

        # fetch combined params
        if structure in (1, 2):
            combined, combined_phi_t, combined_phi_f = combined_all
        else:
            (combined_up, combined_down, combined_phi_t, combined_phi_f
             ) = combined_all

        # compute pad params
        cbs = [(cb[0] if recursive else cb) for cb in combined_all
               if not_none(cb)]
        n_n1s_max = max(len(cb[n2_idx][n1_fr_idx])
                        for cb in cbs
                        for n2_idx in range(len(cb))
                        for n1_fr_idx in range(len(cb[n2_idx])))
        pad_value = 0 if not debug else -2
        left_pad_axis = -2  # left pad along `n1`
        general = False  # use routine optimized for JTFS
        kw = dict(pad_value=pad_value, left_pad_axis=left_pad_axis,
                  general=general)

        # `phi`s #############################################################
        out_phi_t, out_phi_f = None, None
        # ensure `phi`s and spinned pad to the same number of `n1`s
        ref_shape = ((None, None, n_n1s_max, None) if not recursive else
                     (None, None, None, n_n1s_max, None))
        # this will pad along `n1`
        if not_none(combined_phi_t):
            out_phi_t = tensor_padded(combined_phi_t, ref_shape=ref_shape, **kw)
            out_phi_t = process_dims(out_phi_t)
        if not_none(combined_phi_f):
            out_phi_f = tensor_padded(combined_phi_f, ref_shape=ref_shape, **kw)
            out_phi_f = process_dims(out_phi_f)

        # spinned ############################################################
        # don't need `ref_shape` here since by implementation max `n1`s
        # should be found in spinned (`phi`s are trimmed to ensure this)
        if structure in (1, 2):
            out = tensor_padded(combined, **kw)
            out = process_dims(out)

            if structure == 1:
                tp_shape = (0, 2, 1, 3, 4)
                out = B.transpose(out, tp_shape)
                if separate_lowpass:
                    if out_phi_t is not None:
                        out_phi_t = B.transpose(out_phi_t, tp_shape)
                    if out_phi_f is not None:
                        out_phi_f = B.transpose(out_phi_f, tp_shape)

            out = (out if not separate_lowpass else
                   (out, out_phi_f, out_phi_t))

        elif structure in (3, 4):
            out_up   = tensor_padded(combined_up,   **kw)
            out_down = tensor_padded(combined_down, **kw)
            out_up   = process_dims(out_up)
            out_down = process_dims(out_down)

            if structure == 3:
                out = ((out_up, out_down, out_phi_f) if not separate_lowpass else
                       (out_up, out_down, out_phi_f, out_phi_t))
            else:
                if not separate_lowpass:
                    out = (out_up, out_down)
                else:
                    out = (out_up, out_down, out_phi_t)

        # sanity checks ##########################################################
        phis = dict(out_phi_t=out_phi_t, out_phi_f=out_phi_f)
        ref = out[0] if isinstance(out, tuple) else out
        for name, op in phis.items():
            if op is not None:
                errmsg = (name, op.shape, ref.shape)
                # `t`s must match
                assert op.shape[-1] == ref.shape[-1], errmsg
                # number of `n1`s must match
                assert op.shape[-2] == ref.shape[-2], errmsg
                # number of samples must match
                assert op.shape[0]  == ref.shape[0],  errmsg

                # due to transpose
                fr_dim = -3 if structure != 1 else -4
                if 'phi_f' in name:
                    assert op.shape[fr_dim] == 1, op.shape
                    continue

                # compute `ref_fr_len` #######################################
                if structure in (1, 2):
                    ref_fr_len = ref.shape[fr_dim]
                elif structure == 3:
                    # due to separate spins having half of total `n1_fr`s
                    ref_fr_len = ref.shape[fr_dim] * 2
                elif structure == 4:
                    # above + having `psi_t * phi_f`
                    ref_fr_len = (ref.shape[fr_dim] - 1) * 2
                # due to `phi_t * phi_f` being present only in `out_phi_t`
                ref_fr_len = (ref_fr_len if not separate_lowpass else
                              ref_fr_len + 1)

                # assert #####################################################
                assert op.shape[fr_dim] == ref_fr_len, (
                    "{} != {} | {} | {}, {}".format(op.shape[fr_dim], ref_fr_len,
                                                    name, op.shape, ref.shape))
        if structure in (3, 4):
            assert out_up.shape == out_down.shape, (out_up.shape, out_down.shape)

        if not recursive:
            # drop batch dim; `None` in case of `out_exclude`
            if isinstance(out, tuple):
                out = tuple((o[0] if o is not None else o) for o in out)
            else:
                out = out[0]
        return out

    # pack full batch recursively ############################################
    if not isinstance(Scx, dict):
        raise ValueError("must use `out_type` 'dict:array' or 'dict:list' "
                         "for `pack_coeffs_jtfs` (got `type(Scx) = %s`)" % (
                             type(Scx)))

    # infer batch size
    ref_pair = list(Scx)[0]
    if isinstance(Scx[ref_pair], list):
        n_samples = Scx[ref_pair][0]['coef'].shape[0]
    else:  # tensor
        n_samples = Scx[ref_pair].shape[0]
    n_samples = int(n_samples)

    # handle recursion, if applicable
    if n_samples > 1 and sample_idx is None:
        combined_phi_t_s, combined_phi_f_s = [], []
        if structure in (1, 2):
            combined_s = []
        elif structure in (3, 4):
            combined_up_s, combined_down_s = [], []

        for sample_idx in range(n_samples):
            combined_all = pack_coeffs_jtfs(Scx, meta, structure, sample_idx,
                                            separate_lowpass, sampling_psi_fr,
                                            debug, recursive=True)

            combined_phi_t_s.append(combined_all[-2])
            combined_phi_f_s.append(combined_all[-1])
            if structure in (1, 2):
                combined_s.append(combined_all[0])
            elif structure in (3, 4):
                combined_up_s.append(combined_all[0])
                combined_down_s.append(combined_all[1])

        phis = (combined_phi_t_s, combined_phi_f_s)
        if structure in (1, 2):
            combined_all_s = (combined_s, *phis)
        elif structure in (3, 4):
            combined_all_s = (combined_up_s, combined_down_s, *phis)
        out = combined_to_tensor(combined_all_s, recursive=True)
        return out

    ##########################################################################

    # validate `structure` / set default
    structures_available = [1, 2, 3, 4]
    if structure is None:
        structure = structures_available[0]
    elif structure not in structures_available:
        raise ValueError(
            "invalid `structure={}`; Available are: {}".format(
                structure, ','.join(map(str, structures_available))))

    # unpack coeffs for further processing
    Scx_unpacked = {}
    list_coeffs = isinstance(list(Scx.values())[0], list)
    if sample_idx is None and not recursive and n_samples == 1:
        sample_idx = 0

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)
    t_ref = None
    for pair in Scx:
        is_joint = bool(pair not in ('S0', 'S1'))
        if not is_joint:
            continue
        Scx_unpacked[pair] = []
        for coef in Scx[pair]:
            if list_coeffs and (isinstance(coef, dict) and 'coef' in coef):
                coef = coef['coef']
            if t_ref is None:
                t_ref = coef.shape[-1]
            assert coef.shape[-1] == t_ref, (coef.shape, t_ref)

            if coef.ndim == 2:
                Scx_unpacked[pair].extend(coef)
            elif coef.ndim == 1:
                Scx_unpacked[pair].append(coef)
            else:
                raise ValueError("expected `coef.ndim` of 1 or 2, got "
                                 "shape = %s" % str(coef.shape))

    # check that all necessary pairs are present
    pairs = ('psi_t * psi_f_up', 'psi_t * psi_f_down', 'psi_t * phi_f',
             'phi_t * psi_f', 'phi_t * phi_f')
    # structure 4 requires `psi_t * phi_f`
    okay_to_exclude = (pairs[-3:] if structure != 4 else
                       pairs[-2:])
    Scx_pairs = list(Scx)
    for p in pairs:
      if p not in Scx_pairs:
        if (not separate_lowpass or
              (separate_lowpass and p not in okay_to_exclude)):
          raise ValueError(("configuration requires pair '%s', which is "
                            "missing") % p)

    # pack into dictionary indexed by `n1_fr`, `n2` ##########################
    packed = {}
    ns = meta['n']
    n_n1_frs_max = 0
    for pair in pairs:
        if pair not in Scx_pairs:
            continue
        packed[pair] = []
        nsp = ns[pair].astype(int).reshape(-1, 3)

        idx = 0
        n2s_all = nsp[:, 0]
        n2s = np.unique(n2s_all)

        for n2 in n2s:
            packed[pair].append([])
            n1_frs_all = nsp[n2s_all == n2, 1]
            n1_frs = np.unique(n1_frs_all)
            n_n1_frs_max = max(n_n1_frs_max, len(n1_frs))

            for n1_fr in n1_frs:
                packed[pair][-1].append([])
                n1s_done = 0

                if out_3D:
                    # same number of `n1`s for all frequential slices *per-`n2`*
                    n_n1s = len(n1_frs_all)
                    n_n1s_in_n1_fr = n_n1s // len(n1_frs)
                    assert (n_n1s / len(n1_frs)
                            ).is_integer(), (n_n1s, len(n1_frs))
                else:
                    n_n1s_in_n1_fr = len(nsp[n2s_all == n2, 2
                                             ][n1_frs_all == n1_fr])
                if debug:
                    # pack meta instead of coeffs
                    n1s = nsp[n2s_all == n2, 2][n1_frs_all == n1_fr]
                    coef = [[n2, n1_fr, n1, 0] for n1 in n1s]
                    # ensure coef.shape[-1] == t
                    while Scx_unpacked[pair][0].shape[-1] > len(coef[0]):
                        for i in range(len(coef)):
                            coef[i].append(0)
                    coef = np.array(coef)

                    packed[pair][-1][-1].extend(coef)
                    assert len(coef) == n_n1s_in_n1_fr
                    idx += len(coef)
                    n1s_done += len(coef)

                else:
                    while idx < len(nsp) and n1s_done < n_n1s_in_n1_fr:
                        try:
                            coef = Scx_unpacked[pair][idx]
                        except Exception as e:
                            print(pair, idx)
                            raise e
                        if pair == 'phi_t * psi_f':
                            # see "Notes" in docs
                            coef = coef / B.sqrt(2., dtype=coef.dtype)
                        packed[pair][-1][-1].append(coef)
                        idx += 1
                        n1s_done += 1

    # pad along `n1_fr`
    if sampling_psi_fr is None:
        sampling_psi_fr = 'exclude'
    pad_value = 0 if not debug else -2
    for pair in packed:
        if 'psi_f' not in pair:
            continue
        for n2_idx in range(len(packed[pair])):
            if len(packed[pair][n2_idx]) < n_n1_frs_max:
                assert sampling_psi_fr == 'exclude'  # should not occur otherwise
            else:
                continue

            # make a copy to avoid modifying `packed`
            ref = list(tensor_padded(packed[pair][n2_idx][0]))
            # assumes last dim is same (`average=False`)
            # and is 2D, `(n1, t)` (should always be true)
            for i in range(len(ref)):
                if debug:
                    # n2 will be same, everything else variable
                    ref[i][1:] = ref[i][1:] * 0 + pad_value
                else:
                    ref[i] = ref[i] * 0

            while len(packed[pair][n2_idx]) < n_n1_frs_max:
                packed[pair][n2_idx].append(list(ref))

    # pack into list ready to convert to 4D tensor ###########################
    # current indexing: `(n2, n1_fr, n1, time)`
    # reverse every `psi_t` ordering into low-to-high frequency
    combined = packed['psi_t * psi_f_up'][::-1]

    # pack these for later
    # reverse `psi_t` ordering
    combined_down = packed['psi_t * psi_f_down'][::-1]
    # reverse `psi_f` ordering
    for n2 in range(len(combined_down)):
        combined_down[n2] = combined_down[n2][::-1]

    # pack phi_t
    if 'phi_t * psi_f' in Scx_pairs:
        c_phi_t = deepcopy(packed['phi_t * psi_f'])
        c_phi_t[0].append(packed['phi_t * phi_f'][0][0])
        # TODO pointless with `separate_lowpass`
        c_phi_t[0].extend(packed['phi_t * psi_f'][0][::-1])
    else:
        c_phi_t = None
    # pack phi_f
    if 'psi_t * phi_f' in Scx_pairs:
        c_phi_f = packed['psi_t * phi_f'][::-1]
    else:
        c_phi_f = None

    # append `phi_f` if appropriate
    if (not separate_lowpass or structure == 4) and structure != 3:
        # structure 4 joins `psi_t * phi_f` with `psi_t * psi_f`
        for n2_idx in range(len(c_phi_f)):
            for n1_fr_idx in range(len(c_phi_f[n2_idx])):
                c = c_phi_f[n2_idx][n1_fr_idx]
                combined[n2_idx].append(c)
                # not needed in 3, will duplicate in 1 and 2
                if structure == 4:
                    combined_down[n2_idx].insert(0, c)

        l0, l1 = len(combined[0]), len(combined_down[0])
        assert (l0 == l1 if structure == 4 else l0 == l1 + 1), (l0, l1)
    else:
        l0, l1 = len(combined[0]), len(combined_down[0])
        assert l0 == l1, (l0, l1)

    # `None` means they aren't turned to tensors, so only set if need
    # as separate tensors
    combined_phi_t, combined_phi_f = None, None
    if not separate_lowpass:
        if structure == 3:
            combined_phi_f = c_phi_f
    else:
        if structure != 4:
            combined_phi_f = c_phi_f
        combined_phi_t = c_phi_t

    # `3` won't pack `phi_f` with `psi_f` so can't pack `phi_t * phi_f` along
    # `phi_t * psi_f`, must pack with `psi_t * phi_f` instead
    if structure == 3:
        _len = len(c_phi_t[0])
        combined_phi_f.insert(0, c_phi_t[0][_len//2:_len//2 + 1])

    # pack spinned (+ `phi` maybe)
    assert len(combined_down) == len(combined), (
        len(combined_down), len(combined))
    if structure in (1, 2):
        for n2_idx in range(len(combined)):
            combined[n2_idx].extend(combined_down[n2_idx])
        if not separate_lowpass:
            combined.insert(0, c_phi_t[0])
        combined_up = None
    else:
        combined_up = combined
        if not separate_lowpass:
            _len = len(c_phi_t[0])
            # `3` never includes `phi_f` in spinned, so exclude `phi_t * phi_f`
            idx_up   = _len//2 + 1 if structure == 4 else _len//2
            idx_down = _len//2     if structure == 4 else _len//2 + 1
            combined_up.insert(  0, c_phi_t[0][:idx_up])
            combined_down.insert(0, c_phi_t[0][idx_down:])

    # reverse ordering of `n1` ###############################################
    if combined_up is not None:
        cbs = [combined_up, combined_down]
    else:
        cbs = [combined]
    if combined_phi_t is not None:
        cbs.append(combined_phi_t)
    if combined_phi_f is not None:
        cbs.append(combined_phi_f)

    cbs_new = []
    for i, cb in enumerate(cbs):
        cbs_new.append([])
        for n2_idx in range(len(cb)):
            cbs_new[i].append([])
            for n1_fr_idx in range(len(cb[n2_idx])):
                cbs_new[i][n2_idx].append(cb[n2_idx][n1_fr_idx][::-1])

    if combined_up is not None:
        combined_up   = cbs_new.pop(0)
        combined_down = cbs_new.pop(0)
    else:
        combined = cbs_new.pop(0)
    if combined_phi_t is not None:
        combined_phi_t = cbs_new.pop(0)
    if combined_phi_f is not None:
        combined_phi_f = cbs_new.pop(0)
    assert len(cbs_new) == 0, len(cbs_new)

    # finalize ###############################################################
    phis = (combined_phi_t, combined_phi_f)
    combined_all = ((combined, *phis) if combined_up is None else
                    (combined_up, combined_down, *phis))
    if recursive:
        return combined_all
    return combined_to_tensor(combined_all, recursive=False)


def coeff_energy(Scx, meta, pair=None, aggregate=True, correction=False,
                 kind='l2'):
    """Computes energy of JTFS coefficients.

    Current implementation permits computing energy directly via
    `sum(abs(coef)**2)`, hence this method isn't necessary.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose energy to compute.
        If None, will compute for all.

    aggregate: bool (default True)
        True: return one value per `pair`, the sum of all its coeffs' energies
        False: return `(E_flat, E_slices)`, where:
            - E_flat = energy of every coefficient, flattened into a list
              (but still organized pair-wise)
            - E_slices = energy of every joint slice (if not `'S0', 'S1'`),
              in a pair. That is, sum of coeff energies on per-`(n2, n1_fr)`
              basis.

    correction : bool (default False)
        Whether to apply stride and filterbank norm correction factors:
            - stride: if subsampled by 2, energy will reduce by 2
            - filterbank: since input is assumed real, we convolve only over
              positive frequencies, getting half the energy

        Current JTFS implementation accounts for both so default is `False`
        (in fact `True` isn't implemented with any configuration due to
        forced LP sum renormalization - though it's possible to implement).

        Filterbank energy correction is as follows:

            - S0 -> 1
            - U1 -> 2 (because psi_t is only analytic)
            - phi_t * phi_f -> 2 (because U1)
            - psi_t * phi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)
            - phi_t * psi_f -> 4 (because U1 and psi_f is only for one spin)
            - psi_t * psi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)

        For coefficient correction (e.g. distance computation) we instead
        scale the coefficients by square root of these values.

    kind: str['l1', 'l2']
        Kind of energy to compute. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)`
        (so actually L2^2).

    Returns
    -------
    E: float / tuple[list]
        Depends on `pair`, `aggregate`.

    Rationale
    ---------
    Simply `sum(abs(coef)**2)` won't work because we must account for subsampling.

        - Subsampling by `k` will reduce energy (both L1 and L2) by `k`
          (assuming no aliasing).
        - Must account for variable subsampling factors, both in time
          (if `average=False` for S0, S1) and frequency.
          This includes if only seeking ratio (e.g. up vs down), since
          `(a*x0 + b*x1) / (a*y0 + b*y1) != (x0 + x1) / (y0 + y1)`.
    """
    if pair is None or isinstance(pair, (tuple, list)):
        # compute for all (or multiple) pairs
        pairs = pair
        E_flat, E_slices = {}, {}
        for pair in Scx:
            if pairs is not None and pair not in pairs:
                continue
            E_flat[pair], E_slices[pair] = coeff_energy(
                Scx, meta, pair, aggregate=False, kind=kind)
        if aggregate:
            E = {}
            for pair in E_flat:
                E[pair] = np.sum(E_flat[pair])
            return E
        return E_flat, E_slices

    elif not isinstance(pair, str):
        raise ValueError("`pair` must be string, list/tuple of strings, or None "
                         "(got %s)" % pair)

    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: energy(c, kind=kind)
    norm_fn = lambda total_joint_stride: (2**total_joint_stride
                                          if correction else 1)

    E_flat, E_slices = _iterate_coeffs(Scx, meta, pair, fn, norm_fn, factor)
    Es = []
    for s in E_slices:
        Es.append(np.sum(s))
    E_slices = Es

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def coeff_distance(Scx0, Scx1, meta0, meta1=None, pair=None, correction=False,
                   kind='l2'):
    """Computes L2 or L1 relative distance between `Scx0` and `Scx1`.

    Current implementation permits computing distance directly between
    coefficients, as `toolkit.rel_l2(coef0, coef1)`.

    Parameters
    ----------
    Scx0, Scx1: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta0, meta1: dict[dict[np.ndarray]]
        `jtfs.meta()`. If `meta1` is None, will set equal to `meta0`.
        Note that scattering objects responsible for `Scx0` and `Scx1` cannot
        differ in any way that alters coefficient shapes.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose distances to compute.
        If None, will compute for all.

    kind: str['l1', 'l2']
        Kind of distance to compute. L1==`sum(abs(x))`,
        L2==`sqrt(sum(abs(x)**2))`. L1 is not implemented for `correction=False`.

    correction: bool (default False)
        See `help(kymatio.toolkit.coeff_energy)`.

    Returns
    -------
    reldist_flat : list[float]
        Relative distances between individual frequency rows, i.e. per-`n1`.

    reldist_slices : list[float]
        Relative distances between joint slices, i.e. per-`(n2, n1_fr)`.
    """
    if not correction and kind == 'l1':
        raise NotImplementedError

    if meta1 is None:
        meta1 = meta0
    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: c

    def norm_fn(total_joint_stride):
        if not correction:
            return 1
        return (2**(total_joint_stride / 2) if kind == 'l2' else
                2**total_joint_stride)

    c_flat0, c_slices0 = _iterate_coeffs(Scx0, meta0, pair, fn, norm_fn, factor)
    c_flat1, c_slices1 = _iterate_coeffs(Scx1, meta1, pair, fn, norm_fn, factor)

    # make into array and assert shapes are as expected
    c_flat0, c_flat1 = np.asarray(c_flat0), np.asarray(c_flat1)
    c_slices0 = [np.asarray(c) for c in c_slices0]
    c_slices1 = [np.asarray(c) for c in c_slices1]

    assert c_flat0.ndim == c_flat1.ndim == 2, (c_flat0.shape, c_flat1.shape)
    is_joint = bool(pair not in ('S0', 'S1'))
    if is_joint:
        shapes = [np.array(c).shape for cs in (c_slices0, c_slices1) for c in cs]
        # effectively 3D
        assert all(len(s) == 2 for s in shapes), shapes

    E = lambda x: l2(x) if kind == 'l2' else l1(x)
    ref0, ref1 = E(c_flat0), E(c_flat1)
    eps = _eps(ref0, ref1)
    ref = (ref0 + ref1) / 2 + eps

    def D(x0, x1, axis):
        if isinstance(x0, list):
            return [D(_x0, _x1, axis) for (_x0, _x1) in zip(x0, x1)]

        if kind == 'l2':
            return np.sqrt(np.sum(np.abs(x0 - x1)**2, axis=axis)) / ref
        return np.sum(np.abs(x0 - x1), axis=axis) / ref

    # do not collapse `freq` dimension
    reldist_flat   = D(c_flat0,   c_flat1,   axis=-1)
    reldist_slices = D(c_slices0, c_slices1, axis=(-1, -2) if is_joint else -1)

    # return tuple consistency; we don't do "slices" here
    return reldist_flat, reldist_slices


def coeff_energy_ratios(Scx, meta, down_to_up=True, max_to_eps_ratio=10000):
    """Compute ratios of coefficient slice energies, spin down vs up.
    Statistically robust alternative measure to ratio of total energies.

    Parameters
    ----------
    Scx : dict[list] / dict[tensor]
        `jtfs(x)`.

    meta : dict[dict[np.ndarray]]
        `jtfs.meta()`.

    down_to_up : bool (default True)
        Whether to take `E_down / E_up` (True) or `E_up / E_down` (False).
        Note, the actual similarities are opposite, as "down" means convolution
        with down, which is cross-correlation with up.

    max_to_eps_ratio : int
        `eps = max(E_pair0, E_pair1) / max_to_eps_ratio`. Epsilon term
        to use in ratio: `E_pair0 / (E_pair1 + eps)`.

    Returns
    -------
    Ratios of coefficient energies.
    """
    # handle args
    assert isinstance(Scx, dict), ("`Scx` must be dict (got %s); " % type(Scx)
                                   + "set `out_type='dict:array'` or 'dict:list'")

    # compute ratios
    l2s = {}
    pairs = ('psi_t * psi_f_down', 'psi_t * psi_f_up')
    if not down_to_up:
        pairs = pairs[::-1]
    for pair in pairs:
        _, E_slc0 = coeff_energy(Scx, meta, pair=pair, aggregate=False, kind='l2')
        l2s[pair] = np.asarray(E_slc0)

    a, b = l2s.values()
    mx = np.vstack([a, b]).max(axis=0) / max_to_eps_ratio
    eps = np.clip(mx, mx.max() / (max_to_eps_ratio * 1000), None)
    r = a / (b + eps)

    return r


def _get_pair_factor(pair, correction):
    if pair == 'S0' or not correction:
        factor = 1
    elif 'psi' in pair:
        factor = 4
    else:
        factor = 2
    return factor


def _iterate_coeffs(Scx, meta, pair, fn=None, norm_fn=None, factor=None):
    coeffs = drop_batch_dim_jtfs(Scx)[pair]
    out_list = bool(isinstance(coeffs, list))

    # infer out_3D
    if out_list:
        out_3D = bool(coeffs[0]['coef'].ndim == 3)
    else:
        out_3D = bool(coeffs.ndim == 3)

    # fetch backend
    B = ExtendedUnifiedBackend(coeffs)

    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            coeffs_flat.extend(c)
    else:
        if out_3D:
            coeffs = B.reshape(coeffs, (-1, coeffs.shape[-1]))
        coeffs_flat = coeffs

    # prepare for iterating
    meta = deepcopy(meta)  # don't change external dict
    if out_3D:
        meta['stride'][pair] = meta['stride'][pair].reshape(-1, 2)
        meta['n'][pair] = meta['n'][pair].reshape(-1, 3)

    assert (len(coeffs_flat) == len(meta['stride'][pair])), (
        "{} != {} | {}".format(len(coeffs_flat), len(meta['stride'][pair]), pair))

    # define helpers #########################################################
    def get_total_joint_stride(meta_idx):
        n_freqs = 1
        m_start, m_end = meta_idx[0], meta_idx[0] + n_freqs
        stride = meta['stride'][pair][m_start:m_end]
        assert len(stride) != 0, pair

        stride[np.isnan(stride)] = 0
        total_joint_stride = stride.sum()
        meta_idx[0] = m_end  # update idx
        return total_joint_stride

    def n_current():
        i = meta_idx[0]
        m = meta['n'][pair]
        return (m[i] if i <= len(m) - 1 else
                np.array([-3, -3]))  # reached end; ensure equality fails

    def n_is_equal(n0, n1):
        n0, n1 = n0[:2], n1[:2]  # discard U1
        n0[np.isnan(n0)], n1[np.isnan(n1)] = -2, -2  # NaN -> -2
        return bool(np.all(n0 == n1))

    # append energies one by one #############################################
    fn = fn or (lambda c: c)
    norm_fn = norm_fn or (lambda total_joint_stride: 2**total_joint_stride)
    factor = factor or 1

    is_joint = bool(pair not in ('S0', 'S1'))
    E_flat = []
    E_slices = [] if not is_joint else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        if hasattr(c, 'numpy'):
            if hasattr(c, 'cpu') and 'torch' in str(type(c)):
                c = c.cpu()
            c = c.numpy()  # TF/torch
        n_prev = n_current()
        assert c.ndim == 1, (c.shape, pair)
        total_joint_stride = get_total_joint_stride(meta_idx)

        E = norm_fn(total_joint_stride) * fn(c) * factor
        E_flat.append(E)

        if not is_joint:
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices.append([])

    # in case loop terminates early
    if isinstance(E_slices[-1], list) and len(E_slices[-1]) == 0:
        E_slices.pop()

    # ensure they sum to same
    Es_sum = np.sum([np.sum(s) for s in E_slices])
    adiff = abs(np.sum(E_flat) - Es_sum)
    assert np.allclose(np.sum(E_flat), Es_sum), "MAE=%.3f" % adiff

    return E_flat, E_slices


def est_energy_conservation(x, sc=None, T=None, F=None, J=None, J_fr=None,
                            Q=None, Q_fr=None, max_pad_factor=None,
                            max_pad_factor_fr=None, pad_mode=None,
                            pad_mode_fr=None, average=None, average_fr=None,
                            sampling_filters_fr=None, r_psi=None, analytic=None,
                            out_3D=None, aligned=None, jtfs=False, backend=None,
                            verbose=True, get_out=False):
    """Estimate energy conservation given scattering configurations, especially
    scale of averaging. With default settings, passing only `T`/`F`, computes the
    upper bound.

    For time scattering (`jtfs=False`) and non-dyadic length `x`, the estimation
    will be inaccurate per not accounting for energy loss due to unpadding.

    Parameters
    ----------
    x : tensor
        1D input.

    sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
        Scattering object to use. If None, will create per parameters.

    T, F, J, J_fr, Q, Q_fr, max_pad_factor, max_pad_factor_fr, pad_mode,
    pad_mode_fr, average, average_fr, sampling_filters_fr, r_psi, analytic,
    out_3D, aligned:
        Scattering parameters.

    jtfs : bool (default False)
        Whether to estimate per JTFS; if False, does time scattering.
        Must pass also with `sc` to indicate which object it is.
        If `sc` is passed, won't use unaveraged variant where appropriate,
        which won't provide upper bound on energy if `sc(average_fr=True)`.

    backend : None / str
        Backend to use (defaults to torch w/ GPU if available).

    verbose : bool (default True)
        Whether to print results to console.

    get_out : bool (default False)
        Whether to return computed coefficients and scattering objects alongside
        energy ratios.

    Returns
    -------
    ESr : dict[float]
        Energy ratios.

    Scx : tensor / dict[tensor]
        Scattering output (if `get_out==True`).

    sc : `Scattering1D` / `TimeFrequencyScattering1D`
        Scattering object (if `get_out==True`).
    """
    # warn if passing params alongside `sc`
    _kw = dict(T=T, F=F, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr,
               max_pad_factor=max_pad_factor, max_pad_factor_fr=max_pad_factor_fr,
               pad_mode=pad_mode, pad_mode_fr=pad_mode_fr,
               average=average, average_fr=average_fr,
               sampling_filters_fr=sampling_filters_fr,
               out_3D=out_3D, aligned=aligned)
    tm_params = ('T', 'J', 'Q', 'max_pad_factor', 'pad_mode', 'average')
    fr_params = ('F', 'J_fr', 'Q_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                 'average_fr', 'sampling_filters_fr', 'out_3D', 'aligned')
    all_params = (*tm_params, *fr_params)
    if sc is not None and any(_kw[arg] is not None for arg in all_params):
        warnings.warn("`sc` object provided - parametric arguments ignored.")
    elif not jtfs and any(_kw[arg] is not None for arg in fr_params):
        warnings.warn("passed JTFS parameters with `jtfs=False` -- ignored.")

    # create scattering object, if not provided
    if sc is not None:
        if jtfs:
            sc_u = sc_a = sc
    else:
        if jtfs:
            from kymatio import TimeFrequencyScattering1D as SC
        else:
            from kymatio import Scattering1D as SC

        # handle args & pack parameters
        N = x.shape[-1]
        if Q is None:
            Q = (8, 3)
        if pad_mode is None:
            pad_mode = 'reflect'
        if r_psi is None:
            r_psi = (.9, .9, .9) if jtfs else (.9, .9)
        if backend is None:
            try:
                import torch
                backend = 'torch'
            except:
                backend = 'numpy'
        kw = dict(shape=N, J=int(np.log2(N)), T=T, max_pad_factor=max_pad_factor,
                  pad_mode=pad_mode, Q=Q, frontend=backend, r_psi=r_psi)
        if not jtfs:
            if average is None:
                average = True
            if analytic is None:
                analytic = False  # library default
            kw.update(**dict(average=average, analytic=analytic, out_type='list'))
        else:
            # handle `J_fr` & `F`
            if J_fr is None:
                if F is None:
                    sc_temp = SC(**kw)
                    n_psi1 = len(sc_temp.psi1_f)
                    J_fr = int(np.log2(n_psi1)) - 1
                    F = 2**J_fr
                else:
                    J_fr = int(np.log2(F))
            elif F is None:
                F = 2**J_fr
            # handle other args
            if pad_mode_fr is None:
                pad_mode_fr = 'conj-reflect-zero'
            if average_fr is None:
                average_fr = False
            if analytic is None:
                analytic = True  # library default
            if sampling_filters_fr is None:
                sampling_filters_fr = 'resample'
            if Q_fr is None:
                Q_fr = 4

            # pack JTFS args
            kw.update(**dict(max_pad_factor_fr=max_pad_factor_fr, F=F,
                             pad_mode_fr=pad_mode_fr, average_fr=average_fr,
                             analytic=analytic, Q_fr=Q_fr, out_type='dict:list',
                             sampling_filters_fr=sampling_filters_fr,
                             out_3D=out_3D, aligned=aligned))
            if average is None:
                kw_u = dict(**kw, average=False)
                kw_a = dict(**kw, average=True)
            else:
                kw_u = kw_a = dict(**kw, average=average)

        # create scattering object
        if backend == 'torch':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not jtfs:
            sc = SC(**kw)
            if backend == 'torch':
                sc = sc.to(device)
            meta = sc.meta()
        else:
            sc_u, sc_a = SC(**kw_u), SC(**kw_a)
            if backend == 'torch':
                sc_u, sc_a = sc_u.to(device), sc_a.to(device)

    # scatter
    if not jtfs:
        Scx = sc(x)
        Scx = jtfs_to_numpy(Scx)
    else:
        Scx_u = sc_u(x)
        Scx_a = sc_a(x)
        Scx_u, Scx_a = jtfs_to_numpy(Scx_u), jtfs_to_numpy(Scx_a)

    # compute energies
    # input energy
    Ex = energy(x)
    if not jtfs and average:
        Ex /= 2**sc.log2_T

    # scattering energy & ratios
    ES = {}
    if not jtfs:
        for o in (0, 1, 2):
            ES[f'S{o}'] = np.sum([energy(Scx[int(i)]['coef']) for i in
                                  np.where(meta['order'] == o)[0]])
    else:
        for pair in Scx_u:
            Scx = Scx_u if pair not in ('S0', 'S1') else Scx_a
            ES[pair] = np.sum([energy(c['coef']) for c in Scx[pair]])

    ESr = {k: v/Ex for k, v in ES.items()}
    if not jtfs:
        ESr['total'] = np.sum(list(ES.values())) / Ex
    else:
        E_common = sum(ES[pair] for pair in ('S0', 'psi_t * phi_f',
                                             'psi_t * psi_f_up',
                                             'psi_t * psi_f_down'))
        E_v1 = E_common + ES['phi_t * phi_f'] + ES['phi_t * psi_f']
        E_v2 = E_common + ES['S1']
        ESr['total_v1'], ESr['total_v2'] = E_v1 / Ex, E_v2 / Ex

    # print energies
    if T is None:
        T = (sc_a if jtfs else sc).T
    _txt = f", F={F}" if jtfs else ""
    print(f"E(Sx)/E(x) | T={T}{_txt}")
    for k, v in ESr.items():
        print("{:.4f} -- {}".format(v, k))

    if jtfs:
        sc = (sc_u, sc_a)
    return (ESr, Scx, sc) if get_out else ESr


#### Validating 1D filterbank ################################################
def validate_filterbank_tm(sc=None, psi1_f=None, psi2_f=None, phi_f=None,
                           criterion_amplitude=1e-3, verbose=True):
    """Runs `validate_filterbank()` on temporal filters; supports `Scattering1D`
    and `TimeFrequencyScattering1D`.

    Parameters
    ----------
        sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
            If None, then `psi1_f_fr_up`, `psi1_f_fr_down`, and `phi_f_fr` must
            be not None.

        psi1_f : list[tensor] / None
            First-order bandpasses in frequency domain.
            Overridden if `sc` is not None.

        psi2_f : list[tensor] / None
            Second-order bandpasses in frequency domain.
            Overridden if `sc` is not None.

        phi_f : tensor / None
            Lowpass filter in frequency domain.
            Overridden if `sc` is not None.

        criterion_amplitude : float
            Used for various thresholding in `validate_filterbank()`.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data1, data2 : dict, dict
            Returns from `validate_filterbank()` for `psi1_f` and `psi2_f`.
    """
    if sc is None:
        assert not any(arg is None for arg in (psi1_f, psi2_f, phi_f))
    else:
        psi1_f, psi2_f, phi_f = [getattr(sc, k) for k in
                                 ('psi1_f', 'psi2_f', 'phi_f')]
    psi1_f, psi2_f = [[p[0] for p in ps] for ps in (psi1_f, psi2_f)]
    phi_f = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]

    if verbose:
        print("\n// FIRST-ORDER")
    data1 = validate_filterbank(psi1_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    if verbose:
        print("\n\n// SECOND-ORDER")
    data2 = validate_filterbank(psi2_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    return data1, data2


def validate_filterbank_fr(sc=None, psi1_f_fr_up=None, psi1_f_fr_down=None,
                           phi_f_fr=None, j0=0, criterion_amplitude=1e-3,
                           verbose=True):
    """Runs `validate_filterbank()` on frequential filters of JTFS.

    Parameters
    ----------
        sc : `TimeFrequencyScattering1D` / None
            JTFS instance. If None, then `psi1_f_fr_up`, `psi1_f_fr_down`, and
            `phi_f_fr` must be not None.

        psi1_f_fr_up : list[tensor] / None
            Spin up bandpasses in frequency domain.
            Overridden if `sc` is not None.

        psi1_f_fr_down : list[tensor] / None
            Spin down bandpasses in frequency domain.
            Overridden if `sc` is not None.

        phi_f_fr : tensor / None
            Lowpass filter in frequency domain.
            Overridden if `sc` is not None.

        j0 : int
            See `subsample_equiv_due_to_pad` in `help(TimeFrequencyScattering1D)`.

        criterion_amplitude : float
            Used for various thresholding in `validate_filterbank()`.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data_up, data_down : dict, dict
            Returns from `validate_filterbank()` for `psi1_f_fr_up` and
            `psi1_f_fr_down`.
    """
    if sc is None:
        assert not any(arg is None for arg in
                       (psi1_f_fr_up, psi1_f_fr_down, phi_f_fr))
    else:
        psi1_f_fr_up, psi1_f_fr_down, phi_f_fr = [
            getattr(sc, k) for k in
            ('psi1_f_fr_up', 'psi1_f_fr_down', 'phi_f_fr')]
    psi1_f_fr_up, psi1_f_fr_down = [[p[j0] for p in ps] for ps in
                                    (psi1_f_fr_up, psi1_f_fr_down)]
    phi_f_fr = phi_f_fr[j0][0]

    if verbose:
        print("\n// SPIN UP")
    data_up = validate_filterbank(psi1_f_fr_up, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    if verbose:
        print("\n\n// SPIN DOWN")
    data_down = validate_filterbank(psi1_f_fr_down, phi_f_fr, criterion_amplitude,
                                    verbose=verbose,
                                    for_real_inputs=False, unimodal=True)
    return data_up, data_down


def validate_filterbank(psi_fs, phi_f=None, criterion_amplitude=1e-3,
                        for_real_inputs=True, unimodal=True, verbose=True):
    """Checks whether the wavelet filterbank is well-behaved against several
    criterion:

        1. Analyticity:
          - A: Whether analytic *and* anti-analytic filters are present
               (input should contain only one)
          - B: Extent of (anti-)analyticity - whether there's components
               on other side of Nyquist
        2. Aliasing:
          - A. Whether peaks are sorted (left to right or right to left).
               If not, it's possible aliasing (or sloppy user input).
          - B. Whether peaks are distributed exponentially or linearly.
               If neither, it's possible aliasing. (Detection isn't foulproof.)
        3. Zero-mean: whether filters are zero-mean (in time domain)
        4. Zero-phase: whether filters are zero-phase
        5. Frequency coverage: whether filters capture every frequency,
           and whether they do so excessively or insufficiently.
             - Measured with Littlewood-Paley sum (sum of energies).
        6. Redundancy: whether filters overlap excessively (this isn't
           necessarily bad).
             - Measured as ratio of product of energies to sum of energies
               of adjacent filters
        7. Decay:
          - A: Whether any filter is a pure sine (occurs if we try to sample
               a wavelet at too low of a center frequency)
          - B: Whether filters decay sufficiently (in time domain) to avoid
               boundary effects
          B may fail for same reason as 8A & 8B (see these).
        8. Temporal peaks:
          - A: Whether peak is at t==0
          - B: Whether there is only one peak
          - C: Whether decay is smooth (else will incur inflection points)
          A and B may fail to hold for lowest xi due to Morlet's corrective
          term; this is proper behavior.
          See https://www.desmos.com/calculator/ivd7t3mjn8

    Parameters
    ----------
        psi_fs : list[tensor]
            Wavelets in frequency domain, of same length.

        phi_f : tensor
            Lowpass filter in frequency domain, of same length as `psi_fs`.

        criterion_amplitude : float
            Used for various thresholding.

        for_real_inputs : bool (default True)
            Whether the filterbank is intended for real-only inputs.
            E.g. `False` for spinned bandpasses in JTFS.

        unimodal : bool (default True)
            Whether the wavelets have a single peak in frequency domain.
            If `False`, some checks are omitted, and others might be inaccurate.
            Always `True` for Morlet wavelets.

        verbose : bool (default True)
            Whether to print the report.

    Returns
    -------
        data : dict
            Aggregated testing info, along with the report. For keys, see
            `print(list(data))`.
    """
    def pop_if_no_header(report, did_header):
        if not did_header:
            report.pop(-1)

    # fetch basic metadata
    psi_f_0 = psi_fs[0]  # reference filter
    N = len(psi_f_0)

    # assert all inputs are same length
    for n, p in enumerate(psi_fs):
        assert len(p) == N, (len(p), N)
    if phi_f is not None:
        assert len(phi_f) == N, (len(phi_f), N)
    # squeeze into 1D for convenience later
    psi_fs = [p.squeeze() for p in psi_fs]
    if phi_f is not None:
        phi_f = phi_f.squeeze()

    # initialize report
    report = []
    data = {k: {} for k in ('analytic_a_ratio', 'nonzero_mean', 'sine', 'decay',
                            'imag_mean', 'time_peak_idx', 'n_inflections',
                            'redundancy')}
    data['opposite_analytic'] = []

    def title(txt):
        return ("\n== {} " + "=" * (80 - len(txt)) + "\n").format(txt)
    # for later
    w_pos = np.linspace(0, N//2, N//2 + 1, endpoint=True).astype(int)
    w_neg = - w_pos[1:-1][::-1]
    w = np.hstack([w_pos, w_neg])
    eps = np.finfo(psi_f_0.dtype).eps

    peak_idxs = np.array([np.argmax(np.abs(p)) for p in psi_fs])
    peak_idxs_sorted = np.sort(peak_idxs)
    if unimodal and not (np.all(peak_idxs == peak_idxs_sorted) or
                         np.all(peak_idxs == peak_idxs_sorted[::-1])):
        warnings.warn("`psi_fs` peak locations are not sorted; a possible reason "
                      "is aliasing. Will sort, breaking mapping with input's.")
        data['not_sorted'] = True
        peak_idxs = peak_idxs_sorted

    # Analyticity ############################################################
    # check if there are both analytic and anti-analytic bandpasses ##########
    report += [title("ANALYTICITY")]
    did_header = False

    peak_idx_0 = np.argmax(psi_f_0)
    if peak_idx_0 == N // 2:  # ambiguous case; check next filter
        peak_idx_0 = np.argmax(psi_fs[1])
    analytic_0 = bool(peak_idx_0 < N//2)
    # assume entire filterbank is per psi_0
    analyticity = "analytic" if analytic_0 else "anti-analytic"

    for n, p in enumerate(psi_fs[1:]):
        peak_idx_n = np.argmax(p)
        analytic_n = bool(peak_idx_n < N//2)
        if not (analytic_0 is analytic_n):
            if not did_header:
                report += [("Found analytic AND anti-analytic filters in same "
                            "filterbank! psi_fs[0] is {}, but the following "
                            "aren't:\n").format(analyticity)]
                did_header = True
            report += [f"psi_fs[{n}]\n"]
            data['opposite_analytic'].append(n)

    # check if any bandpass isn't strictly analytic/anti- ####################
    did_header = False
    th_ratio = (1 / criterion_amplitude)
    for n, p in enumerate(psi_fs):
        ap = np.abs(p)
        # assume entire filterbank is per psi_0
        if analytic_0:
            # Nyquist is *at* N//2, so to include in sum, index up to N//2 + 1
            a_ratio = (ap[:N//2 + 1].sum() / (ap[N//2 + 1:].sum() + eps))
        else:
            a_ratio = (ap[N//2:].sum() / (ap[:N//2].sum() + eps))
        if a_ratio < th_ratio:
            if not did_header:
                report += [("Found not strictly {} filter(s); threshold for "
                            "ratio of `spectral sum` to `spectral sum past "
                            "Nyquist` is {} - got (less is worse):\n"
                            ).format(analyticity, th_ratio)]
                did_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, a_ratio)]
            data['analytic_a_ratio'][n] = a_ratio

    # check if any bandpass isn't zero-mean ##################################
    pop_if_no_header(report, did_header)
    report += [title("ZERO-MEAN")]
    did_header = False

    for n, p in enumerate(psi_fs):
        if p[0] != 0:
            if not did_header:
                report += ["Found non-zero mean filter(s)!:\n"]
                did_header = True
            report += ["psi_fs[{}][0] == {:.2e}\n".format(n, p[0])]
            data['nonzero_mean'][n] = p[0]

    # Littlewood-Paley sum ###################################################
    def report_lp_sum(report, phi):
        with_phi = not isinstance(phi, int)
        s = "with" if with_phi else "without"
        report += [title("LP-SUM (%s phi)" % s)]
        did_header = False

        lp_sum = lp_sum_psi + np.abs(phi)**2
        lp_sum = (lp_sum[:N//2 + 1] if analytic_0 else
                  lp_sum[N//2:])
        if with_phi:
            data['lp'] = lp_sum
        else:
            data['lp_no_phi'] = lp_sum
        if not with_phi and analytic_0:
            lp_sum = lp_sum[1:]  # exclude dc

        diff_over  = lp_sum - th_lp_sum_over
        diff_under = th_lp_sum_under - lp_sum
        diff_over_max, diff_under_max = diff_over.max(), diff_under.max()
        excess_over  = np.where(diff_over  > th_sum_excess)[0]
        excess_under = np.where(diff_under > th_sum_excess)[0]
        if not analytic_0:
            excess_over  += N//2
            excess_under += N//2
        elif analytic_0 and not with_phi:
            excess_over += 1
            excess_under += 1  # dc

        input_kind = "real" if for_real_inputs else "complex"
        if len(excess_over) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_over) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum exceeds threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_over, input_kind, diff_over_max,
                                 N//2, s)]
            report += ["{}\n\n".format(w[excess_over][::stride])]
            did_header = True
            if with_phi:
                data['lp_excess_over'] = excess_over
                data['lp_excess_over_max'] = diff_over_max
            else:
                data['lp_no_phi_excess_over'] = excess_over
                data['lp_no_phi_excess_over_max'] = diff_over_max

        if len(excess_under) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_under) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum falls below threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse; ~{} implies ~zero "
                        "capturing of the frequency!) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_under, input_kind, diff_under_max,
                                 th_lp_sum_under, N//2, s)]
            # w_show = np.round(w[excess_under][::stride], 3)
            report += ["{}\n\n".format(w[excess_under][::stride])]
            did_header = True
            if with_phi:
                data['lp_excess_under'] = excess_under
                data['lp_excess_under_max'] = diff_under_max
            else:
                data['lp_no_phi_excess_under'] = excess_under
                data['lp_no_phi_excess_under_max'] = diff_under_max

        if did_header:
            stdev = np.abs(lp_sum[lp_sum >= th_lp_sum_under] -
                           th_lp_sum_under).std()
            report += [("Mean absolute deviation from tight frame: {:.2f}\n"
                        "Standard deviation from tight frame: {:.2f} "
                        "(excluded LP sum values below {})\n").format(
                            np.abs(diff_over).mean(), stdev, th_lp_sum_under)]

        pop_if_no_header(report, did_header)

    pop_if_no_header(report, did_header)
    th_lp_sum_over = 2 if for_real_inputs else 1
    th_lp_sum_under = th_lp_sum_over / 2
    th_sum_excess = (1 + criterion_amplitude)**2 - 1
    lp_sum_psi = np.sum([np.abs(p)**2 for p in psi_fs], axis=0)

    # do both cases
    if phi_f is not None:
        report_lp_sum(report, phi=phi_f)
    report_lp_sum(report, phi=0)

    # Redundancy #############################################################
    from .scattering1d.filter_bank import compute_filter_redundancy

    report += [title("REDUNDANCY")]
    did_header = False
    th_r = .4 if for_real_inputs else .2

    max_to_print = 20
    printed = 0
    for n in range(len(psi_fs) - 1):
        r = compute_filter_redundancy(psi_fs[n], psi_fs[n + 1])
        data['redundancy'][(n, n + 1)] = r
        if r > th_r:
            if not did_header:
                report += [("Found filters with redundancy exceeding {} (energy "
                            "overlap relative to sum of individual energies) "
                            "-- This isn't necessarily bad. Showing up to 20 "
                            "filters:\n").format(th_r)]
                did_header = True
            report += ["psi_fs[{}] & psi_fs[{}]: {:.3f}\n".format(n, n + 1, r)]
            printed += 1
            if printed >= max_to_print:
                break

    # Decay: check if any bandpass is a pure sine ############################
    pop_if_no_header(report, did_header)
    report += [title("DECAY (check for pure sines)")]
    did_header = False
    th_ratio_max_to_next_max = (1 / criterion_amplitude)

    for n, p in enumerate(psi_fs):
        psort = np.sort(np.abs(p))  # least to greatest
        ratio = psort[-1] / (psort[-2] + eps)
        if ratio > th_ratio_max_to_next_max:
            if not did_header:
                report += [("Found filter(s) that are pure sines! Threshold for "
                            "ratio of Fourier peak to next-highest value is {} "
                            "- got (more is worse):\n"
                            ).format(th_ratio_max_to_next_max)]
                did_header = True
            report += ["psi_fs[{}]: {:.2e}\n".format(n, ratio)]
            data['sine'][n] = ratio

    # Decay: boundary effects ################################################
    pop_if_no_header(report, did_header)
    report += [title("DECAY (boundary effects)")]
    did_header = False
    th_ratio_max_to_min = (1 / criterion_amplitude)

    psis = [np.fft.ifft(p) for p in psi_fs]
    apsis = [np.abs(p) for p in psis]
    for n, ap in enumerate(apsis):
        ratio = ap.max() / (ap.min() + eps)
        if ratio < th_ratio_max_to_min:
            if not did_header:
                report += [("Found filter(s) with incomplete decay (will incur "
                            "boundary effects), with following ratios of "
                            "amplitude max to edge (less is worse; threshold "
                            "is {}):\n").format(1 / criterion_amplitude)]
                did_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, ratio)]
            data['decay'][n] = ratio

    # check lowpass
    if phi_f is not None:
        aphi = np.abs(np.fft.ifft(phi_f))
        ratio = aphi.max() / (aphi.min() + eps)
        if ratio < th_ratio_max_to_min:
            nl = "\n" if did_header else ""
            report += [("{}Lowpass filter has incomplete decay (will incur "
                        "boundary effects), with following ratio of amplitude "
                        "max to edge: {:.1f} > {}\n").format(nl, ratio,
                                                             th_ratio_max_to_min)]
            did_header = True
            data['decay'][-1] = ratio

    # Phase ##################################################################
    pop_if_no_header(report, did_header)
    report += [title("PHASE")]
    did_header = False
    th_imag_mean = eps

    for n, p in enumerate(psi_fs):
        imag_mean = np.abs(p.imag).mean()
        if imag_mean > th_imag_mean:
            if not did_header:
                report += [("Found filters with non-zero phase, with following "
                            "absolute mean imaginary values:\n")]
            report += ["psi_fs[{}]: {:.1e}\n".format(n, imag_mean)]
            data['imag_mean'][n] = imag_mean

    # Aliasing ###############################################################
    def diff_extend(diff, th, cond='gt', order=1):
        # the idea is to take `diff` without losing samples, if the goal is
        # `where(diff == 0)`; `diff` is forward difference, and two samples
        # participated in producing the zero, where later one's index is dropped
        # E.g. detecting duplicate peak indices:
        # [0, 1, 3, 3, 5] -> diff gives [2], so take [2, 3]
        # but instead of adding an index, replace next sample with zero such that
        # its `where == 0` produces that index
        if order > 1:
            diff_e = diff_extend(diff, th)
            for o in range(order - 1):
                diff_e = diff_e(diff_e, th)
            return diff_e

        diff_e = []
        d_extend = 2*th if cond == 'gt' else th
        prev_true = False
        for d in diff:
            if prev_true:
                diff_e.append(d_extend)
                prev_true = False
            else:
                diff_e.append(d)
            if (cond == 'gt' and np.abs(d) > th or
                cond == 'eq' and np.abs(d) == th):
                prev_true = True
        if prev_true:
            # last sample was zero; extend
            diff_e.append(d_extend)
        return np.array(diff_e)

    if unimodal:
        pop_if_no_header(report, did_header)
        report += [title("ALIASING")]
        did_header = False
        eps_big = eps * 100  # ease threshold for "zero"

        # check whether peak locations follow a linear or exponential
        # distribution, progressively dropping those that do to see if any remain

        # x[n] = A^n + C; x[n] - x[n - 1] = A^n - A^(n-1) = A^n*(1 - A) = A^n*C
        # log(A^n*C) = K + n; diff(diff(K + n)) == 0
        # `abs` for anti-analytic case with descending idxs
        logdiffs = np.diff(np.log(np.abs(np.diff(peak_idxs))), 2)
        # In general it's impossible to determine whether a rounded sequence
        # samples an exponential, since if the exponential rate (A in A^n) is
        # sufficiently small, its rounded values will be linear over some portion.
        # However, it cannot be anything else, and we are okay with linear
        # (unless constant, i.e. duplicate, captured elsewhere) - thus the net
        # case of `exp + lin` is still captured. The only uncertainty is in
        # the threshold; assuming deviation by at most 1 sample, we set it to 1.
        # A test is:
        # `for b in linspace(1.2, 6.5, 500): x = round(b**arange(10) + 50)`
        # with `if any(abs(diff, o).min() == 0 for o in (1, 2, 3)): continue`,
        # Another with: `linspace(.2, 1, 500)` and `round(256*b**arange(10) + 50)`
        # to exclude `x` with repeated or linear values
        # However, while this has no false positives (never misses an exp/lin),
        # it can also count some non-exp/lin as exp/lin, but this is rare.
        # To be safe, per above test, we use the empirical value of 0.9
        logdiffs_extended = diff_extend(logdiffs, .9)
        if len(logdiffs_extended) > len(logdiffs) + 2:
            # this could be `assert` but not worth erroring over this
            warnings.warn("`len(logdiffs_extended) > len(logdiffs) + 2`; will "
                          "use more conservative estimate on peaks distribution")
            logdiffs_extended = logdiffs
        keep = np.where(np.abs(logdiffs_extended) > .9)
        # note due to three `diff`s we artificially exclude 3 samples
        peak_idxs_remainder = peak_idxs[keep]

        # now constant (diff_order==1) and linear (diff_order==2)
        for diff_order in (1, 2):
            idxs_diff2 = np.diff(peak_idxs_remainder, diff_order)
            keep = np.where(np.abs(idxs_diff2) > eps_big)
            peak_idxs_remainder = peak_idxs_remainder[keep]

        # if anything remains, it's neither
        if len(peak_idxs_remainder) > 0:
            report += [("Found Fourier peaks that are spaced neither "
                        "exponentially nor linearly, suggesting possible "
                        "aliasing.\npsi_fs[n], n={}\n"
                        ).format(peak_idxs_remainder)]
            data['alias_peak_idxs'] = peak_idxs_remainder
            did_header = True

        # check if any peaks repeat ##########################################
        # e.g. [0, 1, 5, 5, 3] -> diff gives [3], so take [3, 4]
        peak_idxs_zeros = diff_extend(np.diff(peak_idxs), th=0, cond='eq')
        peak_idxs_zeros = np.where(peak_idxs_zeros == 0)[0]
        if len(peak_idxs_zeros) > 0:
            nl = "\n" if did_header else ""
            report += ["{}Found duplicate Fourier peaks!:\n{}\n".format(
                nl, peak_idxs_zeros)]
            did_header = True

    # Temporal peak ##########################################################
    if unimodal:
        # check that temporal peak is at t==0 ################################
        pop_if_no_header(report, did_header)
        report += [title("TEMPORAL PEAK")]
        did_header = False

        for n, ap in enumerate(apsis):
            peak_idx = np.argmax(ap)
            if peak_idx != 0:
                if not did_header:
                    report += [("Found filters with temporal peak not at t=0!, "
                                "with following peak locations:\n")]
                    did_header = True
                report += ["psi_fs[{}]: {}\n".format(n, peak_idx)]
                data['time_peak_idx'][n] = peak_idx

        # check that there is only one temporal peak #########################
        pop_if_no_header(report, did_header)
        did_header = False
        for n, ap in enumerate(apsis):
            # count number of inflection points (where sign of derivative changes)
            # exclude very small values
            # center for proper `diff`
            ap = np.fft.ifftshift(ap)
            inflections = np.diff(np.sign(np.diff(ap[ap > 10*eps])))
            n_inflections = sum(np.abs(inflections) > eps)

            if n_inflections > 1:
                if not did_header:
                    report += [("\nFound filters with multiple temporal peaks "
                                "(or incomplete/non-smooth decay)! "
                                "(more precisely, >1 inflection points) with "
                                "following number of inflection points:\n")]
                    did_header = True
                report += ["psi_fs[{}]: {}\n".format(n, n_inflections)]
                data['n_inflections'] = n_inflections

    # Print report ###########################################################
    pop_if_no_header(report, did_header)
    report = ''.join(report)
    data['report'] = report
    if verbose:
        if len(report) == 0:
            print("Perfect filterbank!")
        else:
            print(report)
    return data


#### energy & distance #######################################################
def energy(x, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    x = x['coef'] if isinstance(x, dict) else x
    B = ExtendedUnifiedBackend(x)
    out = (B.norm(x, ord=1) if kind == 'l1' else
           B.norm(x, ord=2)**2)
    if np.prod(out.shape) == 1:
        out = float(out)
    return out


def l2(x, axis=None, keepdims=True):
    """`sqrt(sum(abs(x)**2))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=2, axis=axis, keepdims=keepdims)

def rel_l2(x0, x1, axis=None, adj=False):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = l2(x0, axis) if not adj else (l2(x0, axis) + l2(x1, axis)) / 2
    return l2(x1 - x0, axis) / ref


def l1(x, axis=None, keepdims=True):
    """`sum(abs(x))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=1, axis=axis, keepdims=keepdims)

def rel_l1(x0, x1, adj=False, axis=None):
    ref = l1(x0, axis) if not adj else (l1(x0, axis) + l1(x1, axis)) / 2
    return l1(x1 - x0, axis) / ref


def rel_ae(x0, x1, eps=None, ref_both=True):
    """Relative absolute error."""
    B = ExtendedUnifiedBackend(x0)
    if ref_both:
        if eps is None:
            eps = _eps(x0, x1)
        ref = (x0 + x1)/2 + eps
    else:
        if eps is None:
            eps = _eps(x0)
        ref = x0 + eps
    return B.abs(x0 - x1) / ref


def _eps(x0, x1=None):
    B = ExtendedUnifiedBackend(x0)
    if x1 is None:
        eps =  B.abs(x0).max() / 1000
    else:
        eps = (B.abs(x0).max() + B.abs(x1).max()) / 2000
    eps = max(eps, 10 * np.finfo(B.numpy(x0).dtype).eps)
    return eps

#### test signals ###########################################################
def echirp(N, fmin=1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    phi = _echirp_fn(fmin, fmax, tmin, tmax)(t)
    return np.cos(phi)


def _echirp_fn(fmin, fmax, tmin=0, tmax=1):
    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)
    phi = lambda t: 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return phi


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None,
         partials_f_sep=1.6, endpoint=False):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=endpoint)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    pad_right = (N - len(window)) // 2
    pad_left = N - len(window) - pad_right
    window = np.pad(window, [pad_left, pad_right])

    x = np.zeros(N)
    xs = x.copy()
    for p in range(0, n_partials):
        f_shift = partials_f_sep**p
        x_partial = np.sin(2*np.pi * f0 * f_shift * t) * window
        partial_shift = int(total_shift * np.log2(f_shift) / np.log2(n_partials))
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial
    return x, xs

#### misc ###################################################################
def tensor_padded(seq, pad_value=0, init_fn=None, cast_fn=None, ref_shape=None,
                  left_pad_axis=None, general=True):
    """Make tensor from variable-length `seq` (e.g. nested list) padded with
    `fill_value`.

    Parameters
    ----------
    seq : list[tensor]
        Nested list of tensors.

    pad_value : float
        Value to pad with.

    init_fn : function / None
        Instantiates packing tensor, e.g. `lambda shape: torch.zeros(shape)`.
        Defaults to `backend.full`.

    cast_fn : function / None
        Function to cast tensor values before packing, e.g.
        `lambda x: torch.tensor(x)`.

    ref_shape : tuple[int] / None
        Tensor output shape to pack into, instead of what's inferred for `seq`,
        as long as >= that shape. Shape inferred from `seq` is the minimal size
        determined from longest list at each nest level, i.e. we can't go lower
        without losing elements.

        Tuple can contain `None`: `(None, 3)` will pad dim0 per `seq` and dim1
        to 3, unless `seq`'s dim1 is 4, then will pad to 4.

        Recommended to pass this argument if applying `tensor_padded` multiple
        times, as time to infer shape is significant, especially relative to
        GPU computation.

    left_pad_axis : int / tuple[int] / None
        Along these axes, will pad from left instead of the default right.
        Not implemented for dim0 (`0 in left_pad_axis`).

    general : bool (default True)
        If `False`, will use a much faster routine that's for JTFS.

    Not implemented for TensorFlow: will convert to numpy array then rever to
    TF tensor.

    Code borrows from: https://stackoverflow.com/a/27890978/10133797
    """
    iter_axis = [0]
    prev_axis = [iter_axis[0]]

    def fill_tensor(arr, seq, fill_value=0):
        if iter_axis[0] != prev_axis[0]:
            prev_axis[0] = iter_axis[0]

        if arr.ndim == 1:
            try:
                len_ = len(seq)
            except TypeError:
                len_ = 0

            if len_ == 0:
                pass
            elif len(shape) not in left_pad_axis:  # right pad
                arr[:len_] = cast_fn(seq)
            else:  # left pad
                arr[-len_:] = cast_fn(seq)
        else:
            iter_axis[0] += 1

            left_pad = bool(iter_axis[0] in left_pad_axis)
            if left_pad:
                seq = IterWithDelay(seq, len(arr) - len(seq), fillvalue=())

            for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
                fill_tensor(subarr, subseq, fill_value)
                if subarr.ndim != 1:
                    iter_axis[0] -= 1

    # infer `init_fn` and `cast_fn` from `seq`, if not provided ##############
    backend, backend_name = _infer_backend(seq, get_name=True)
    is_tf = bool(backend_name == 'tensorflow')
    if is_tf:
        tf = backend
        backend = np
        backend_name = 'numpy'

    # infer dtype & whether on GPU
    sq = seq
    while isinstance(sq, list):
        sq = sq[0]
    dtype = sq.dtype if hasattr(sq, 'dtype') else type(sq)
    if backend_name == 'torch':
        device = sq.device
    else:
        device = None

    if init_fn is None:
        if backend_name == 'numpy':
            if is_tf:
                dtype = dtype.name
            init_fn = lambda s: np.full(s, pad_value, dtype=dtype)
        elif backend_name == 'torch':
            init_fn = lambda s: backend.full(s, pad_value, dtype=dtype,
                                             device=device)

    if cast_fn is None:
        if is_tf:
            cast_fn = lambda x: x.numpy()
        elif backend_name == 'numpy':
            cast_fn = lambda x: x
        elif backend_name == 'torch':
            cast_fn = lambda x: (backend.tensor(x, device=device)
                                 if not isinstance(x, backend.Tensor) else x)

    ##########################################################################
    # infer shape if not provided
    if ref_shape is None or any(s is None for s in ref_shape):
        shape = list(find_shape(seq, fast=not general))
        # override shape with `ref_shape` where provided
        if ref_shape is not None:
            for i, s in enumerate(ref_shape):
                if s is not None and s >= shape[i]:
                    shape[i] = s
    else:
        shape = ref_shape
    shape = tuple(shape)

    # handle `left_pad_axis`
    if left_pad_axis is None:
        left_pad_axis = ()
    elif isinstance(left_pad_axis, int):
        left_pad_axis = [left_pad_axis]
    elif isinstance(left_pad_axis, tuple):
        left_pad_axis = list(left_pad_axis)
    # negatives -> positives
    for i, a in enumerate(left_pad_axis):
        if a < 0:
            # +1 since counting index depth which goes `1, 2, ...`
            left_pad_axis[i] = (len(shape) + 1) + a
    if 0 in left_pad_axis:
        raise NotImplementedError("`0` in `left_pad_axis`")

    # fill
    arr = init_fn(shape)
    fill_tensor(arr, seq, fill_value=pad_value)

    # revert if needed
    if is_tf:
        arr = tf.convert_to_tensor(arr)
    return arr


def find_shape(seq, fast=False):
    """Finds shape to pad a variably nested list to.
    `fast=True` uses an implementation optimized for JTFS.
    """
    if fast:
        """Assumes 4D/5D and only variable length lists are in dim3."""
        flat = chain.from_iterable
        try:
            dim4 = len(seq[0][0][0][0])  # 5D
            dims = (len(seq), len(seq[0]), len(seq[0][0]),
                    max(map(len, flat(flat(seq)))), dim4)
        except:
            dims = (len(seq), len(seq[0]),
                    max(map(len, flat(seq))), len(seq[0][0][0]))
    else:
        dims = _find_shape_gen(seq)
    return dims


def _find_shape_gen(seq):
    """Code borrows from https://stackoverflow.com/a/27890978/10133797"""
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [_find_shape_gen(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in
                           zip_longest(*shapes, fillvalue=1))


class IterWithDelay():
    """Allows implementing left padding by delaying iteration of a sequence."""
    def __init__(self, x, delay, fillvalue=()):
        self.x = x
        self.delay = delay
        self.fillvalue = fillvalue

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx > self.delay - 1:
            idx = self.idx - self.delay
            if idx < len(self.x):
                out = self.x[idx]
            else:
                raise StopIteration
        else:
            out = self.fillvalue
        self.idx += 1
        return out


# backend ####################################################################
class ExtendedUnifiedBackend():
    """Extends existing Kymatio backend with functionality."""
    def __init__(self, x_or_backend_name):
        if isinstance(x_or_backend_name, str):
            backend_name = x_or_backend_name
        else:
            backend_name = _infer_backend(x_or_backend_name, get_name=True)[1]
        self.backend_name = backend_name
        if backend_name == 'torch':
            import torch
            self.B = torch
        elif backend_name == 'tensorflow':
            import tensorflow as tf
            self.B = tf
        else:
            self.B = np
        self.Bk = get_kymatio_backend(backend_name)

    def __getattr__(self, name):
        # fetch from kymatio.backend if possible
        if hasattr(self.Bk, name):
            return getattr(self.Bk, name)
        raise AttributeError(f"'{self.Bk.__name__}' object has no "
                             f"attribute '{name}'")

    def abs(self, x):
        return self.B.abs(x)

    def log(self, x):
        if self.backend_name == 'numpy':
            out = np.log(x)
        elif self.backend_name == 'torch':
            out = self.B.log(x)
        else:
            out = self.B.math.log(x)
        return out

    def sum(self, x, axis=None, keepdims=False):
        if self.backend_name == 'numpy':
            out = np.sum(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.sum(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.reduce_sum(x, axis=axis, keepdims=keepdims)
        return out

    def norm(self, x, ord=2, axis=None, keepdims=True):
        if self.backend_name == 'numpy':
            if ord == 1:
                out = np.sum(np.abs(x), axis=axis, keepdims=keepdims)
            elif ord == 2:
                out = np.linalg.norm(x, ord=None, axis=axis, keepdims=keepdims)
            else:
                out = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.norm(x, p=ord, dim=axis, keepdim=keepdims)
        else:
            out = self.B.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        return out

    def median(self, x, axis=None, keepdims=None):
        if keepdims is None and self.backend_name != 'tensorflow':
            keepdims = True
        if self.backend_name == 'numpy':
            out = np.median(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.median(x, dim=axis, keepdim=keepdims)
        else:
            if axis is not None or keepdims is not None:
                raise ValueError("`axis` and `keepdims` for `median` in "
                                 "TensorFlow backend are not implemented.")
            v = self.B.reshape(x, [-1])
            m = v.get_shape()[0]//2
            out = self.B.reduce_min(self.B.nn.top_k(v, m, sorted=False).values)
        return out

    def std(self, x, axis=None, keepdims=True):
        if self.backend_name == 'numpy':
            out = np.std(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.std(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.math.reduce_std(x, axis=axis, keepdims=keepdims)
        return out

    def min(self, x, axis=None, keepdims=False):
        if self.backend_name == 'numpy':
            out = np.min(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.min(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.math.reduce_min(x, axis=axis, keepdims=keepdims)
        return out

    def numpy(self, x):
        if self.backend_name == 'numpy':
            out = x
        else:
            if hasattr(x, 'to') and 'cpu' not in x.device.type:
                x = x.cpu()
            if getattr(x, 'requires_grad', False):
                x = x.detach()
            out = x.numpy()
        return out


def _infer_backend(x, get_name=False):
    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]

    module = type(x).__module__.split('.')[0]

    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    elif isinstance(x, (int, float)):
        # list of lists, fallback to numpy
        module = 'numpy'
        backend = np
    else:
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))


def get_kymatio_backend(backend_name):
    if backend_name == 'numpy':
        from .backend.numpy_backend import NumpyBackend as B
    elif backend_name == 'torch':
        from .backend.torch_backend import TorchBackend as B
    elif backend_name == 'tensorflow':
        from .backend.tensorflow_backend import TensorFlowBackend as B
    return B
