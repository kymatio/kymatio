# -*- coding: utf-8 -*-
"""Convenience visual methods."""
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['gif_jtfs', 'filterbank_scattering']


def gif_jtfs(Scx, meta, norms=None, inf_token=-1, skip_spins=False):
    """Slice heatmaps of Joint Time-Frequency Scattering.

    # Arguments:
        Scx: np.ndarray
            JTFS.

        meta: list[dict]
            `scattering.meta()`.

        norms: None / tuple
            Plot color norms for 1) `psi_t * psi_f`, 2) `psi_t * phi_f`, and
            3) `phi_t * psi_f` pairs, respectively.
            Tuple of three (upper limits only, lower assumed 0).
            If None, will auto-norm each slice (not recommended).
            # TODO make better default

        inf_token: int / np.nan
            Placeholder used in `meta` to denote infinity.

        skip_spins: bool (default False)
            Whether to skip `psi_t * psi_f` pairs.

    # Example:
        T, J, Q = 2049, 7, 16
        x = np.cos(np.pi * 350 ** np.linspace(0, 1, T))

        scattering = TimeFrequencyScattering(J, T, Q, J_fr=4, Q_fr=2,
                                             out_type='list', average=True)
        Scx = scattering(x)
        meta = scattering.meta()

        gif_jtfs(Scx, meta)
    """
    def _title(meta, i, spin):
        txt = r"$|\Psi_{%s, %s, %s} \star X|$"
        mu, l = [int(n) if (float(n).is_integer() and n >= 0) else '-\infty'
                 for n in meta['n'][i]]
        return (txt % (mu, l, spin), {'fontsize': 20})

    def _viz_spins(Scx, meta, i, norm, n_spins):
        i0, i1 = i, i + n_spins

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        kw = dict(abs=1, ticks=0, show=0, norm=norm)
        imshow(Scx[i0]['coef'], ax=axes[0], **kw, title=_title(meta, i0, '+1'))
        imshow(Scx[i1]['coef'], ax=axes[1], **kw, title=_title(meta, i1, '-1'))
        plt.subplots_adjust(wspace=0.01)
        plt.show()

    def _viz_simple(Scx, meta, i, norm):
        imshow(Scx[i]['coef'], abs=1, ticks=0, show=1, norm=norm, w=.8, h=.5,
               title=_title(meta, i, '0'))

    if norms is not None:
        norms = [(0, n) for n in norms]
    else:
        norms = (None, None, None)
    n_spins = sum(int(s) for s in meta['s'][1:] if s == 1)

    i = 0
    # skip first-order
    while meta['n'][i][1] != 0:
        i += 1
    i += 1

    # `psi_t * psi_f` first
    while inf_token not in meta['n'][i + n_spins]:
        if not skip_spins:
            _viz_spins(Scx, meta, i, norms[0], n_spins)
        i += 1
    i += n_spins

    # `psi_t * phi_f`
    while meta['n'][i][1] == inf_token:
        _viz_simple(Scx, meta, i, norms[1])
        i += 1

    # `phi_t * psi_f`
    while meta['n'][i][0] == inf_token:
        _viz_simple(Scx, meta, i, norms[2])
        i += 1
        if i >= len(Scx):
            break


#### Visuals primitives ## messy code ########################################
def imshow(x, title=None, show=True, cmap=None, norm=None, abs=0,
           w=None, h=None, ticks=True, aspect='auto', ax=None, fig=None,
           yticks=None, xticks=None, xlabel=None, ylabel=None, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if norm is None:
        mx = np.max(np.abs(x))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    if cmap is None:
        cmap = 'jet' if abs else 'bwr'
    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        ax.imshow(np.abs(x), **_kw)
    else:
        ax.imshow(x.real, **_kw)

    if w or h:
        fig.set_size_inches(12 * (w or 1), 12 * (h or 1))

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks)
    if xlabel is not None:
        ax.set_xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        ax.set_ylabel(ylabel, weight='bold', fontsize=15)

    if title is not None:
        _title(title, ax=ax)
    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, complex=0, abs=0, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         xlabel=None, ylabel=None, xticks=None, yticks=None, ticks=True,
         ax=None, fig=None, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)

    # styling
    if vlines:
        vhlines(vlines, kind='v')
    if hlines:
        vhlines(hlines, kind='h')
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks)

    if title is not None:
        _title(title, ax=ax)

    # tighten plot boundaries
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin
    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)

    # xlims/ylims, labels
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)

    # size, display
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if show:
        plt.show()


def vhlines(lines, kind='v'):
    lfn = plt.axvline if kind=='v' else plt.axhline

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)

    for line in lines:
        lfn(line, **lkw)


def _ticks(xticks, yticks):
    def fmt(ticks):
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.2f")

    if yticks is not None:
        idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
        yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
        plt.yticks(idxs, yt)
    if xticks is not None:
        idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
        xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
        plt.xticks(idxs, xt)


def _title(title, ax=None):
    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    defaults = dict(loc='left', fontsize=17, weight='bold')
    for k, v in defaults.items():
        kw[k] = kw.get(k, v)

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)


def filterbank_scattering(scattering, zoom=0, second_order=False):
    """
    # Arguments:
        scattering: kymatio.scattering1d.Scattering1D
            Scattering object.
        zoom: int
            Will zoom plots by this many octaves.
            If -1, will show full frequency axis (including negatives).
        second_order: bool (default False)
            Whether to plot second-order wavelets.

    # Example:
        scattering = Scattering1D(shape=2048, J=8, Q=8)
        filterbank_scattering(scattering)
    """
    def _plot_filters(ps, p0, title):
        # Morlets
        for p in ps:
            j = p['j']
            plot(p[0], color=colors[j], linestyle=linestyles[j])

        # octave bounds
        Nmax = len(p[0])
        plot([], vlines=([Nmax//2**j for j in range(scattering.J)],
                         dict(color='k', linewidth=1)))
        # lowpass
        if zoom == -1:
            xlims = (-.02 * Nmax, 1.02 * Nmax)
        else:
            xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)
        plot(p0[0], color='k', xlims=xlims, title=title, show=1)

    # define colors & linestyles
    colors = [f"tab:{c}" for c in ("blue orange green red purple brown pink "
                                   "gray olive cyan".split())]
    linestyles = ('-', '--', '-.')
    nc = len(colors)
    nls = len(linestyles)

    # support J up to nc * nls
    colors = colors * nls
    linestyles = [ls_set for ls in "- -- -.".split() for ls_set in [ls]*nc]

    # shorthand references
    p0 = scattering.phi_f
    p1 = scattering.psi1_f
    p2 = scattering.psi2_f

    _plot_filters(p1, p0, title="First-order filterbank")

    if second_order:
        _plot_filters(p2, p0, title="Second-order filterbank")
