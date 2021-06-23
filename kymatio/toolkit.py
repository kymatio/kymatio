# -*- coding: utf-8 -*-
"""Convenience utilities."""
import numpy as np
from copy import deepcopy


def drop_batch_dim_jtfs(Scx):
    """Drop dim 0 of every JTFS coefficient (if it's `1`).

    Doesn't modify input:
        - dict/list: new list/dict (with copied meta if applicable)
        - array: new object but shared storage with original array (so original
          variable reference points to unsqueezed array).
    """
    def get_meta(s):
        return {k: v for k, v in s.items() if not hasattr(v, 'ndim')}

    backend = _infer_backend(Scx)
    if isinstance(Scx, dict):
        out = {}  # don't modify source dict
        for pair in Scx:
            out[pair] = []
            if isinstance(Scx[pair], list):
                for i, s in enumerate(Scx[pair]):
                    out[pair].append(get_meta(s))
                    out[pair][i]['coef'] = backend.squeeze(s['coef'], 0)
            else:
                out[pair] = backend.squeeze(Scx[pair], 0)
    elif isinstance(Scx, list):
        out = []  # don't modify source list
        for s in Scx:
            o = get_meta(s)
            o['coef'] = backend.squeeze(s['coef'], 0)
            out.append(o)
    elif isinstance(Scx, tuple):  # out_type=='array' && out_3D==True
        Scx = (backend.squeeze(Scx[0], 0),
               backend.squeeze(Scx[1], 0))
    elif hasattr(Scx, 'ndim'):
        Scx = backend.squeeze(Scx, 0)
    else:
        raise ValueError(("unrecognized input type: {}; must be as returned by "
                          "`jtfs(x)`.").format(type(Scx)))
    return out


def coeff_energy(Scx, meta, pair=None, aggregate=True, kind='l2'):
    """Computes energy of JTFS coefficients.

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

    # compute compensation factor # TODO better comment
    factor = _get_pair_factor(pair)
    fn = lambda c: energy(c, kind=kind)
    norm_fn = lambda total_joint_stride: 2**total_joint_stride

    E_flat, E_slices = _iterate_coeffs(Scx, meta, pair, fn, norm_fn, factor)
    Es = []
    for s in E_slices:
        Es.append(np.sum(s))
    E_slices = Es

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def coeff_distance(Scx0, Scx1, meta0, meta1=None, pair=None, kind='l2'):
    if meta1 is None:
        meta1 = meta0
    # compute compensation factor # TODO better comment
    factor = _get_pair_factor(pair)
    fn = lambda c: c

    def norm_fn(total_joint_stride):
        return (2**(total_joint_stride / 2) if kind == 'l2' else
                2**total_joint_stride)

    c_flat0, c_slices0 = _iterate_coeffs(Scx0, meta0, pair, fn, norm_fn, factor)
    c_flat1, c_slices1 = _iterate_coeffs(Scx1, meta0, pair, fn, norm_fn, factor)

    # make into array and assert shapes are as expected
    c_flat0, c_flat1 = np.asarray(c_flat0), np.asarray(c_flat1)
    c_slices0 = [np.asarray(c) for c in c_slices0]
    c_slices1 = [np.asarray(c) for c in c_slices1]

    assert c_flat0.ndim == c_flat1.ndim == 2, (c_flat0.shape, c_flat1.shape)
    shapes = [np.array(c).shape for cs in (c_slices0, c_slices1) for c in cs]
    # effectively 3D
    assert all(len(s) == 2 for s in shapes), shapes

    E = lambda x: _l2(x) if kind == 'l2' else _l1(x)
    ref0, ref1 = E(c_flat0), E(c_flat1)
    ref = (ref0 + ref1) / 2

    def D(x0, x1, axis):
        if isinstance(x0, list):
            return [D(_x0, _x1, axis) for (_x0, _x1) in zip(x0, x1)]

        if kind == 'l2':
            return np.sqrt(np.sum(np.abs(x0 - x1)**2, axis=axis)) / ref
        return np.sum(np.abs(x0 - x1), axis=axis) / ref

    # do not collapse `freq` dimension
    reldist_flat   = D(c_flat0,   c_flat1,   axis=-1)
    reldist_slices = D(c_slices0, c_slices1, axis=(-1, -2))

    # return tuple consistency; we don't do "slices" here
    return reldist_flat, reldist_slices


def _get_pair_factor(pair):
    if pair == 'S0':
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
    out_3D = bool(meta['stride'][pair].ndim == 3)
    if out_3D:
        assert coeffs.ndim == 3, coeffs.ndim
    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            coeffs_flat.extend(c)
    else:
        if out_3D:
            coeffs = coeffs.reshape(-1, coeffs.shape[-1])
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

    E_flat = []
    E_slices = [] if pair in ('S0', 'S1') else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        if hasattr(c, 'numpy'):
            if hasattr(c, 'cpu'):  # torch
                c = c.cpu()
            c = c.numpy()  # TF/torch
        n_prev = n_current()
        assert c.ndim == 1, (c.shape, pair)
        total_joint_stride = get_total_joint_stride(meta_idx)

        E = norm_fn(total_joint_stride) * fn(c) * factor
        E_flat.append(E)

        if pair in ('S0', 'S1'):
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices.append([])

    # # in case loop terminates early
    if isinstance(E_slices[-1], list) and len(E_slices[-1]) == 0:
        E_slices.pop()

    # ensure they sum to same
    Es_sum = np.sum(np.sum(E_slices))
    adiff = abs(np.sum(E_flat) - Es_sum)
    assert np.allclose(np.sum(E_flat), Es_sum), "MAE=%.3f" % adiff

    return E_flat, E_slices


def energy(x, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    def sum(x):
        if type(x).__module__ == 'tensorflow':
            return backend.reduce_sum(x)
        return backend.sum(x)

    x = x['coef'] if isinstance(x, dict) else x
    backend = _infer_backend(x)
    return (backend.sum(backend.abs(x)) if kind == 'l1' else
            backend.sum(backend.abs(x)**2))


def _l2(x):  # TODO backend # TODO use `energy` instead?
    return np.sqrt(np.sum(np.abs(x)**2))

def l2(x0, x1, adj=False):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = _l2(x0) if not adj else (_l2(x0) + _l2(x1)) / 2
    return _l2(x1 - x0) / ref


def _l1(x):
    return np.sum(np.abs(x))

def l1(x0, x1, adj=False):
    ref = _l1(x0) if not adj else (_l1(x0) + _l1(x1)) / 2
    return _l1(x1 - x0) / ref


def _infer_backend(x, get_name=False):
    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]
    module = type(x).__module__
    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    else:
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))
