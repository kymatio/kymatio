# -*- coding: utf-8 -*-
"""Convenience utilities."""
import numpy as np
import scipy.signal


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
    """
    pass


def coeff_energy(Scx, meta, pair=None, aggregate=True, correction=False,
                 kind='l2'):
    """Computes energy of JTFS coefficients -- dummified."""
    out = {}
    if pair is None:
        for pair in Scx:
            out[pair] = energy(Scx[pair])
    else:
        if not isinstance(pair, (tuple, list)):
            pair = [pair]
        for p in Scx:
            if p in pair:
                out[p] = energy(Scx[p])
    return out


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
        E_slc0 = coeff_energy(Scx, meta, pair=pair, aggregate=False, kind='l2')
        l2s[pair] = np.asarray(E_slc0[pair])

    a, b = l2s.values()
    mx = np.vstack([a, b]).max(axis=0) / max_to_eps_ratio
    eps = np.clip(mx, mx.max() / (max_to_eps_ratio * 1000), None)
    r = a / (b + eps)

    return r


#### energy & distance #######################################################
def energy(x, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    x = x['coef'] if isinstance(x, dict) else x
    B = ExtendedUnifiedBackend(x)
    out = (B.norm(x, ord=1) if kind == 'l1' else
           B.norm(x, ord=2)**2)
    if np.sum(out.size) == 1:
        out = float(out)
    return out


def l2(x, axis=None):
    """`sqrt(sum(abs(x)**2))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=2, axis=axis)

def rel_l2(x0, x1, axis=None, adj=False):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = l2(x0, axis) if not adj else (l2(x0, axis) + l2(x1, axis)) / 2
    return l2(x1 - x0, axis) / ref


def l1(x, axis=None):
    """`sum(abs(x))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=1)

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
