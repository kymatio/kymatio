# -*- coding: utf-8 -*-
"""Convenience visual methods."""
import numpy as np
import matplotlib.pyplot as plt
from .scattering1d.filter_bank import compute_temporal_support
from scipy.fft import ifft


__all__ = ['gif_jtfs', 'filterbank_scattering', 'filterbank_jtfs',
           'energy_profile_jtfs']


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
        plot([], vlines=([Nmax//2**j for j in range(1, scattering.J + 2)],
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

    title = "First-order filterbank | J, Q1 = {}, {}".format(
        scattering.J, scattering.Q[0])
    _plot_filters(p1, p0, title=title)

    if second_order:
        title = "Second-order filterbank | J, Q2 = {}, {}".format(
            scattering.J, scattering.Q[1])
        _plot_filters(p2, p0, title=title)


def filterbank_jtfs(jtfs, part='real', zoomed=False):
    """
    # Arguments:
        jtfs: kymatio.scattering1d.TimeFrequencyScattering1D
            JTFS instance.
        part: str['real', 'imag', 'complex']
            Whether to plot real or imaginary part (black-white-red colormap),
            or complex (special coloring).
        zoomed: bool (default False)
            Whether to plot all filters with maximum subsampling
            (loses relative orientations but shows fine detail).

    # Examples:
        T, J, Q, J_fr, Q_fr = 512, 5, 16, 3, 1
        jtfs = TimeFrequencyScattering1D(J, T, Q, J_fr=J_fr, Q_fr=Q_fr,
                                         out_type='array', average_fr=1)
        filterbank_jtfs(jtfs)
    """
    def to_time(p):
        # center & ifft
        return ifft(p[0] * (-1)**np.arange(len(p[0])))

    def get_imshow_data(jtfs, t_idx, f_idx, s_idx=None, max_t_bound=None,
                        max_f_bound=None):
        # if lowpass, get lowpass data #######################################
        if t_idx == -1 and f_idx == -1:
            p_t, p_f = jtfs.phi_f, jtfs.sc_freq.phi_f
            psi_txt = r"$\Psi_{%s, %s, %s}$" % ("-\infty", "-\infty", 0)
        elif t_idx == -1:
            p_t, p_f = jtfs.phi_f, jtfs.sc_freq.psi1_f_up[f_idx]
            psi_txt = r"$\Psi_{%s, %s, %s}$" % ("-\infty", f_idx, 0)
        elif f_idx == -1:
            p_t, p_f = jtfs.psi2_f[t_idx], jtfs.sc_freq.phi_f
            psi_txt = r"$\Psi_{%s, %s, %s}$" % (t_idx, "-\infty", 0)

        if t_idx == -1 or f_idx == -1:
            title = (psi_txt, dict(fontsize=17, y=.75))
            Psi = to_time(p_f)[:, None] * to_time(p_t)[None]
            if max_t_bound is not None or zoomed:
                Psi = process_Psi(Psi, max_t_bound, max_f_bound)
            return Psi, title

        # else get spinned wavelets ##########################################
        psi_spin = (jtfs.sc_freq.psi1_f_up if s_idx == 0 else
                    jtfs.sc_freq.psi1_f_down)
        psi_f = psi_spin[f_idx]
        psi_t = jtfs.psi2_f[t_idx]

        f_width = compute_temporal_support(psi_f[0][None])
        t_width = compute_temporal_support(psi_t[0][None])
        f_bound = int(2**np.floor(np.log2(f_width)))
        t_bound = int(2**np.floor(np.log2(t_width)))

        # to time
        psi_f = to_time(psi_f)
        psi_t = to_time(psi_t)

        # compute joint wavelet in time
        # TODO fix up & down instead of conjugating
        Psi = np.conj(psi_f)[:, None] * psi_t[None]
        # title
        spin = '+1' if s_idx == 0 else '-1'
        psi_txt = r"$\Psi_{%s, %s, %s}$" % (t_idx, f_idx, spin)
        title = (psi_txt, dict(fontsize=17, y=.75))
        # meta
        m = dict(t_bound=t_bound, f_bound=f_bound)
        return Psi, title, m

    def process_Psi(Psi, max_t_bound, max_f_bound, m=None):
        M, N = Psi.shape
        if zoomed and m is not None:
            f_bound, t_bound = m['f_bound'], m['t_bound']
        else:
            f_bound, t_bound = max_f_bound, max_t_bound

        # ensure doesn't exceed own or max bounds
        Psi = Psi[max(0, M//2 - f_bound):min(M, M//2 + f_bound),
                  max(0, N//2 - t_bound):min(N, N//2 + t_bound)]
        if zoomed:
            return Psi

        # pad to common size if too short
        f_diff = 2*f_bound - Psi.shape[0]
        t_diff = 2*t_bound - Psi.shape[1]
        Psi = np.pad(Psi, [[int(np.ceil(f_diff/2)), f_diff//2],
                           [int(np.ceil(t_diff/2)), t_diff//2]])
        return Psi

    def _show(Psi, title, ax):
        if part == 'real':
            Psi = Psi.real
        elif part == 'imag':
            Psi = Psi.imag
        else:
            Psi = _colorize_complex(Psi)
        cmap = 'bwr' if part in ('real', 'imag') else 'none'
        imshow(Psi, title=title, show=0, ax=ax, ticks=0, borders=0, cmap=cmap)

    # get spinned wavelet arrays & metadata ##################################
    n_rows, n_cols = len(jtfs.psi2_f), len(jtfs.sc_freq.psi1_f_up)
    imshow_data = {}
    for s_idx in (0, 1):
        for t_idx in range(n_rows):
            for f_idx in range(n_cols):
                imshow_data[(s_idx, t_idx, f_idx)
                            ] = get_imshow_data(jtfs, t_idx, f_idx, s_idx)

    max_t_bound = max(data[2]['t_bound'] for data in imshow_data.values())
    max_f_bound = max(data[2]['f_bound'] for data in imshow_data.values())
    bounds = (max_t_bound, max_f_bound)

    # plot ###################################################################
    fig, axes = plt.subplots(n_rows * 2 + 1, n_cols + 1, figsize=(12, 13))
    _txt = "(%s%s" % (part, " part" if part in ('real', 'imag') else "")
    _txt += ", zoomed)" if zoomed else ")"
    plt.suptitle("Joint wavelet filterbank " + _txt,
                 y=1.04, weight='bold', fontsize=17)

    # (psi_t * psi_f) and (phi_t * psi_f)
    for s_idx in (0, 1):
        # (phi_t * psi_f)
        if s_idx == 1 and t_idx == n_rows - 1 and f_idx == n_cols - 1:
            for f_idx in range(n_cols):
                Psi, title  = get_imshow_data(jtfs, -1, f_idx, 0, *bounds)
                if zoomed:
                    Psi = process_Psi(Psi, None, None,
                                      imshow_data[(0, 0, f_idx)][2])
                row_idx = n_rows
                ax = axes[row_idx][f_idx]
                _show(Psi, title=title, ax=ax)

        # (psi_t * psi_f)
        for t_idx in range(n_rows):
            for f_idx in range(n_cols):
                psi_t_idx = t_idx if s_idx == 0 else (n_rows - 1 - t_idx)
                Psi, title, m = imshow_data[(s_idx, psi_t_idx, f_idx)]
                Psi = process_Psi(Psi, max_t_bound, max_f_bound, m)

                row_idx = t_idx + (n_rows + 1) * s_idx
                ax = axes[row_idx][f_idx]
                _show(Psi, title=title, ax=ax)

    # (psi_t * phi_f)
    for t_idx in range(n_rows):
        Psi, title = get_imshow_data(jtfs, t_idx, -1, 0, *bounds)
        if zoomed:
            Psi = process_Psi(Psi, None, None,
                              imshow_data[(0, t_idx, 0)][2])
        ax = axes[t_idx][-1]
        _show(Psi, title=title, ax=ax)

    # (phi_t * phi_f)
    Psi, title = get_imshow_data(jtfs, -1, -1, 0, *bounds)
    if zoomed:
        Psi = process_Psi(Psi, None, None,
                          imshow_data[(0, 0, 0)][2])
    ax = axes[n_rows][-1]
    _show(Psi, title=title, ax=ax)

    # strip borders of remainders
    for t_idx in range(n_rows + 1, 2*n_rows + 1):
        ax = axes[t_idx][-1]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    # tight
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)


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
            If None, will norm to `.5 * max(coeffs)`, where coeffs = all joint
            coeffs except `phi_t * phi_f`.

        inf_token: int / np.nan
            Placeholder used in `meta` to denote infinity.

        skip_spins: bool (default False)
            Whether to skip `psi_t * psi_f` pairs.

    # Example:
        T, J, Q = 2049, 7, 16
        x = np.cos(np.pi * 350 ** np.linspace(0, 1, T))

        scattering = TimeFrequencyScattering1D(J, T, Q, J_fr=4, Q_fr=2,
                                               out_type='list', average=True)
        Scx = scattering(x)
        meta = scattering.meta()

        gif_jtfs(Scx, meta)
    """
    def _title(meta, i, pair, spin):
        txt = r"$|\Psi_{%s, %s, %s} \star X|$"
        mu, l = [int(n) if (float(n).is_integer() and n >= 0) else '-\infty'
                 for n in meta['n'][pair][i]]
        return (txt % (mu, l, spin), {'fontsize': 20})

    def _viz_spins(Scx, meta, i, norm):
        kup = 'psi_t * psi_f_up'
        kdn = 'psi_t * psi_f_down'
        sup, sdn = Scx[kup][i], Scx[kdn][i]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        kw = dict(abs=1, ticks=0, show=0, norm=norm)

        imshow(sup, ax=axes[0], **kw, title=_title(meta, i, kup, +1))
        imshow(sdn, ax=axes[1], **kw, title=_title(meta, i, kdn, -1))
        plt.subplots_adjust(wspace=0.01)
        plt.show()

    def _viz_simple(coef, pair, meta, i, norm):
        imshow(coef, abs=1, ticks=0, show=1, norm=norm, w=.8, h=.5,
               title=_title(meta, i, pair, '0'))

    if norms is not None:
        norms = [(0, n) for n in norms]
    else:
        # set to .5 times the max of any joint coefficient (except phi_t * phi_f)
        mx = np.max([(c['coef'] if isinstance(c, list) else c).max()
                     for pair in Scx for c in Scx[pair]
                     if pair not in ('S0', 'S1', 'phi_t * phi_f')])
        norms = [(0, .5 * mx)] * 5

    if not skip_spins:
        for i in range(len(Scx['psi_t * psi_f_up'])):
            _viz_spins(Scx, meta, i, norms[0])

    pairs = ('psi_t * phi_f', 'phi_t * psi_f', 'phi_t * phi_f')
    for j, pair in enumerate(pairs):
        for i, coef in enumerate(Scx[pair]):
            coef = coef['coef'] if isinstance(coef, list) else coef
            _viz_simple(coef, pair, meta, i, norms[1 + j])


def energy_profile_jtfs(Scx, log2_T, log2_F, x=None, kind='L2'):
    """Plot & print relevant energy information across coefficient pairs.

    Parameters
    ----------
    Scx : dict
        Output of `TimeFrequencyScattering1D.scattering()`.
    log2_T : int
        `TimeFrequencyScattering1D.log2_T`. `average=False`    -> `log2_T=1`.
    log2_F : int
        `TimeFrequencyScattering1D.log2_T`. `average_fr=False` -> `log2_F=1`.
    x : tensor, optional
        Original input to print `E_out / E_in`.
    kind : str['L1', 'L2']
        - L1: sum(abs(x))
        - L2: sum(abs(x)**2) -- actually L2^2
    """
    def energy(x):
        return (np.abs(x).sum() if kind == 'L1' else
                (np.abs(x)**2).sum())

    # TODO reverse coeff ordering low to high freq
    # extract energy info
    energies = []
    pair_energies = {}
    idxs = []
    # enforce pair order
    pairs = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_down')
    for pair in pairs:
        norm = 2**log2_T if pair in ('S0', 'S1') else 2**(log2_T + log2_F)
        E = [norm * energy(c['coef'] if isinstance(c, list) else c)
             for c in Scx[pair]]
        pair_energies[pair] = np.sum(E)
        energies.extend(E)
        idxs.append(len(energies) - 1)

    # format & plot ##########################################################
    energies = np.array(energies)
    ticks = np.arange(len(energies))
    vlines = (idxs, {'color': 'tab:red', 'linewidth': 1})

    scat(ticks[idxs], energies[idxs], s=20)
    title = "{} | S0, S1, phi_t * phi_f, phi_t *, * phi_f, up, down".format(
        "L1 norm" if kind == 'L1' else "Energy")
    plot(energies, vlines=vlines, show=1, title=title)

    # cumulative sum
    energies_cs = np.cumsum(energies)
    scat(ticks[idxs], energies_cs[idxs], s=20)
    plot(energies_cs, ylims=(0, None), vlines=vlines, show=1,
         title="cumsum(%s)" % ("L1" if kind == 'L1' else "Energy"))

    # print report ###########################################################
    e_total = np.sum(energies)
    nums = ["%.1f" % pair_energies[pair] for pair in pair_energies]
    longest_num = max(map(len, nums))

    for i, pair in enumerate(pairs):
        e_perc = "%.1f" % (np.sum(pair_energies[pair]) / e_total * 100)
        print("{} ({}%) -- {}".format(
            nums[i].ljust(longest_num), str(e_perc).rjust(4), pair))

    # E_out / E_in
    if x is not None:
        print("E_out / E_in = %.3f" % (e_total / energy(x)))


#### Visuals primitives ## messy code ########################################
def imshow(x, title=None, show=True, cmap=None, norm=None, abs=0,
           w=None, h=None, ticks=True, borders=True, aspect='auto',
           ax=None, fig=None, yticks=None, xticks=None, xlabel=None, ylabel=None,
           **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
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
    if cmap == 'none':
        cmap = None
    elif cmap is None:
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
    if not borders:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
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
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel)


def scat(x, y=None, title=None, show=0, s=18, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None, ticks=1,
         complex=False, abs=False, xlabel=None, ylabel=None, ax=None, fig=None,
         **kw):
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
        ax.scatter(x, y.real, s=s, **kw)
        ax.scatter(x, y.imag, s=s, **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.scatter(x, y, s=s, **kw)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        _title(title, ax=ax)
    if vlines:
        vhlines(vlines, kind='v')
    if hlines:
        vhlines(hlines, kind='h')
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel)


def vhlines(lines, kind='v'):
    lfn = getattr(plt, f'ax{kind}line')

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


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, xlabel=None, ylabel=None):
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin

    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)
    if show:
        plt.show()


def _colorize_complex(z):
    """Map complex `z` to 3D array suitable for complex image visualization.

    Borrowed from https://stackoverflow.com/a/20958684/10133797
    """
    from colorsys import hls_to_rgb
    z = z / np.abs(z).max()
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 / (1 + r)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)
    c = np.array(c)
    c = c.swapaxes(0,2)
    return c
