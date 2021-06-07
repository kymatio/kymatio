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

    kind: str['l2', 'l2']
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
            for p in E_flat:
                E[pair] = E_flat[pair].sum()
            return E
        return E_flat, E_slices

    elif not isinstance(pair, str):
        raise ValueError("`pair` must be string, list/tuple of strings, or None "
                         "(got %s)" % pair)

    # compute for one pair ###################################################
    coeffs = drop_batch_dim_jtfs(Scx)[pair]
    out_list = bool(isinstance(coeffs, list))

    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            coeffs_flat.extend(c)
    else:
        out_3D = bool(coeffs.ndim == 3)
        if out_3D:
            coeffs = coeffs.reshape(-1, coeffs.shape[-1])
        coeffs_flat = coeffs

    # prepare for iterating
    meta = deepcopy(meta)  # don't change external dict
    out_3D = bool(meta['stride'][pair].ndim == 3)
    if out_3D:
        meta['stride'][pair] = meta['stride'][pair].reshape(-1, 2)
        meta['n'][pair] = meta['n'][pair].reshape(-1, 3)

    # define helpers #########################################################
    def norm(c, meta_idx):
        n_freqs = 1
        m_start, m_end = meta_idx[0], meta_idx[0] + n_freqs
        stride = meta['stride'][pair][m_start:m_end]
        assert c.ndim == 1, (c.shape, pair)

        stride[np.isnan(stride)] = 0
        total_joint_stride = stride.sum()
        norm = 2 ** total_joint_stride
        meta_idx[0] = m_end  # update idx
        return norm

    def n_current():
        i = meta_idx[0]
        m = meta['n'][pair]
        return (m[i] if i in m else
                m[i - 1])  # reached end

    def n_is_equal(n0, n1):
        n0, n1 = n0[:2], n1[:2]  # discard U1
        n0[np.isnan(n0)], n1[np.isnan(n1)] = -2, -2  # NaN -> -2
        return bool(np.all(n0 == n1))

    # append energies one by one #############################################
    E_flat = []
    E_slices = [] if pair in ('S0', 'S1') else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        n_prev = n_current()
        E = norm(c, meta_idx) * energy(c)
        E_flat.append(E)

        if pair in ('S0', 'S1'):
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices[-1] = np.sum(E_slices[-1])  # aggregate slice
            E_slices.append([])

    # in case loop terminates early
    if isinstance(E_slices[-1], list):
        E_slices[-1] = np.sum(E_slices[-1])

    # ensure they sum to same
    adiff = abs(np.sum(E_flat) - np.sum(E_slices))
    assert np.allclose(np.sum(E_flat), np.sum(E_slices)), "MAE=%.3f" % adiff

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def energy(x, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    x = x['coef'] if isinstance(x, dict) else x
    return (np.abs(x).sum() if kind == 'l2' else
            (np.abs(x)**2).sum())


def _infer_backend(x):
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
        return np
    elif module == 'torch':
        import torch
        return torch
    elif module == 'tensorflow':
        import tensorflow
        return tensorflow
    else:
        raise ValueError("could not infer backend from %s" % type(x))
