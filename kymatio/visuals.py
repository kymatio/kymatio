# -*- coding: utf-8 -*-
"""Convenience visual methods."""
import os
import numpy as np
from scipy.fft import ifft, ifftshift
from copy import deepcopy
from .scattering1d.filter_bank import compute_temporal_support
from .toolkit import coeff_energy, coeff_distance, energy, drop_batch_dim_jtfs

try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("`kymatio.visuals` requires `matplotlib` installed.")


def filterbank_scattering(scattering, zoom=0, filterbank=True, lp_sum=False,
                          lp_phi=True, first_order=True, second_order=False,
                          plot_kw=None):
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
    def _plot_filters(ps, p0, lp, title):
        # determine plot parameters ##########################################
        # vertical lines (octave bounds)
        Nmax = len(ps[0][0])
        # x-axis zoom
        if zoom == -1:
            xlims = (-.02 * Nmax, 1.02 * Nmax)
        else:
            xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)

        if 'xlims' not in user_plot_kw_names:
            plot_kw['xlims'] = xlims
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = title

        # plot filterbank ####################################################
        if filterbank:
            # Morlets
            for p in ps:
                j = p['j']
                plot(p[0], color=colors[j], linestyle=linestyles[j])
            # vlines
            plot([], vlines=([Nmax//2**j for j in range(1, scattering.J + 2)],
                             dict(color='k', linewidth=1)))
            # lowpass
            if isinstance(p0[0], list):
                p0 = p0[0]
            plot(p0[0], color='k', **plot_kw, show=1)

        # plot LP sum ########################################################
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = "Littlewood-Paley sum"
        if lp_sum:
            plot(lp, **plot_kw, show=1,
                 hlines=(2, dict(color='tab:red', linestyle='--')),
                 vlines=(Nmax//2, dict(color='k', linewidth=1)))

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)

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
    if second_order:
        p2 = scattering.psi2_f

    # compute LP sum
    lp1, lp2 = 0, 0
    if lp_sum:
        # it's list for JTFS
        p0_longest = p0[0] if not isinstance(p0[0], list) else p0[0][0]
        for p in p1:
            lp1 += np.abs(p[0])**2
        if lp_phi:
            lp1 += np.abs(p0_longest)**2

        if second_order:
            for p in p2:
                lp2 += np.abs(p[0])**2
            if lp_phi:
                lp2 += np.abs(p0_longest)**2

    # title & plot
    if first_order:
        title = "First-order filterbank | J, Q1, T = {}, {}, {}".format(
            scattering.J, scattering.Q[0], scattering.T)
        _plot_filters(p1, p0, lp1, title=title)

    if second_order:
        title = "Second-order filterbank | J, Q2, T = {}, {}, {}".format(
            scattering.J, scattering.Q[1], scattering.T)
        _plot_filters(p2, p0, lp2, title=title)


def filterbank_jtfs_1d(jtfs, zoom=0, j0=0, filterbank=True, lp_sum=False,
                       lp_phi=True, center_dc=None, plot_kw=None):
    """Visualize JTFS frequential filterbank.

    Parameters
    ----------
    scattering: kymatio.scattering1d.TimeFrequencyScattering1D
        Scattering object.

    zoom: int
        Will zoom plots by this many octaves.
        If -1, will show full frequency axis (including negatives),
        and both spins.

    j0: int
        `subsample_equiv_due_to_pad`

    filterbank : bool (default True)
        Whether to plot the filterbank.

    lp_sum: bool (default False)
        Whether to plot Littlewood-Paley sum of the filterbank.

    center_dc: bool / None
        If True, will `ifftshift` to center the dc bin.
        Defaults to `True` if `zoom == -1`.

    plot_kw: None / dict
        Will pass to `plot(**plot_kw)`.

    Example
    -------
    ::

        scattering = Scattering1D(shape=2048, J=8, Q=8)
        filterbank_scattering(scattering)
    """
    def _plot_filters(ps, p0, lp, title_base, up):
        # determine plot parameters ##########################################
        # vertical lines (octave bounds)
        Nmax = len(ps[0][j0])
        j_dists = np.array([Nmax//2**j for j in range(1, jtfs.J_fr + 1)])
        if up:
            vlines = (Nmax//2 + j_dists if center_dc else
                      Nmax - j_dists)
        else:
            vlines = (Nmax//2 - j_dists if center_dc else
                      j_dists)
        # x-axis zoom
        if zoom == -1:
            xlims = (-.02 * Nmax, 1.02 * Nmax)
        else:
            xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)
            if up:
                xlims = (Nmax - xlims[1], Nmax - .2 * xlims[0])

        # title
        show = (zoom != -1) or not up
        if zoom != -1:
            title = title_base % "up" if up else title_base % "down"
        else:
            title = title_base

        # handle `plot_kw`
        if 'xlims' not in user_plot_kw_names:
            plot_kw['xlims'] = xlims
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = title

        # plot filterbank ####################################################
        if filterbank:
            # bandpasses
            for p in ps:
                if j0 not in p:  # sampling_psi_fr == 'exclude'
                    continue
                j = p['j'][j0]
                pplot = p[j0].squeeze()
                if center_dc:
                    pplot = ifftshift(pplot)
                    pplot[1:] = pplot[1:][::-1]
                plot(pplot, color=colors[j], linestyle=linestyles[j])
            # lowpass
            p0plot = p0[j0][0].squeeze()
            if center_dc:
                p0plot = ifftshift(p0plot)
                p0plot[1:] = p0plot[1:][::-1]
            plot(p0plot, color='k', **plot_kw, show=show,
                 vlines=(vlines, dict(color='k', linewidth=1)))

        # plot LP sum ########################################################
        plot_kw_lp = {}
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = ("Littlewood-Paley sum" +
                                " (no phi)" * int(not lp_phi))
        if 'ylims' not in user_plot_kw_names:
            plot_kw_lp['ylims'] = (0, None)
        if lp_sum and not (zoom == -1 and up):
            lpplot = ifftshift(lp) if center_dc else lp
            plot(lpplot, **plot_kw, **plot_kw_lp, show=show,
                 hlines=(1, dict(color='tab:red', linestyle='--')),
                 vlines=(Nmax//2, dict(color='k', linewidth=1)))

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)
    # handle `center_dc`
    if center_dc is None:
        center_dc = bool(zoom == -1)

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
    p0 = jtfs.sc_freq.phi_f_fr
    pup = jtfs.psi1_f_fr_up
    pdn = jtfs.psi1_f_fr_down

    # compute LP sum
    lp = 0
    if lp_sum:
        for psi1_f in (pup, pdn):
            for p in psi1_f:
                if j0 not in p:  # sampling_psi_fr == 'exclude'
                    continue
                lp += np.abs(p[j0])**2
        if lp_phi:
            lp += np.abs(p0[j0][0])**2

    # title
    params = (jtfs.J_fr, jtfs.Q_fr, jtfs.F)
    if zoom != -1:
        title_base = ("Frequential filterbank | spin %s | J_fr, Q_fr, F = "
                      "{}, {}, {}").format(*params)
    else:
        title_base = ("Frequential filterbank | J_fr, Q_fr, F = "
                      "{}, {}, {}").format(*params)

    # plot
    _plot_filters(pup, p0, lp, title_base=title_base, up=True)
    _plot_filters(pdn, p0, lp, title_base=title_base, up=False)


def filterbank_jtfs(jtfs, part='real', zoomed=False, w=1, h=1, borders=False,
                    labels=True, suptitle_y=1.015):
    """Visualize JTFS joint 2D filterbank in time domai.

    Parameters
    ----------
    jtfs: kymatio.scattering1d.TimeFrequencyScattering1D
        JTFS instance.

    part: str['real', 'imag', 'complex']
        Whether to plot real or imaginary part (black-white-red colormap),
        or complex (special coloring).

    zoomed: bool (default False)
        Whether to plot all filters with maximum subsampling
        (loses relative orientations but shows fine detail).

    w, h: float, float
        Adjust width and height.

    borders: bool (default False)
        Whether to show plot borders between wavelets.

    labels: bool (default True)
        Whether to label joint slices with `mu, l, spin` information.

    suptitle_y: float / None
        Position of plot title (None for no title).
        Default is optimized for `w=h=1`.

    Example
    -------
    ::

        T, J, Q, J_fr, Q_fr = 512, 5, 16, 3, 1
        jtfs = TimeFrequencyScattering1D(J, T, Q, J_fr=J_fr, Q_fr=Q_fr,
                                         out_type='array', average_fr=1)
        filterbank_jtfs(jtfs)
    """
    def to_time(p):
        # center & ifft
        return ifft(p * (-1)**np.arange(len(p)))

    def get_imshow_data(jtfs, t_idx, f_idx, s_idx=None, max_t_bound=None,
                        max_f_bound=None):
        # iterate freq wavelets backwards to arrange low-high freq <-> left-right
        _f_idx = len(jtfs.psi1_f_fr_up) - f_idx - 1
        # if lowpass, get lowpass data #######################################
        if t_idx == -1 and f_idx == -1:
            p_t, p_f = jtfs.phi_f[0], jtfs.sc_freq.phi_f_fr[0]
            psi_txt = r"$\Psi_{%s, %s, %s}$" % ("-\infty", "-\infty", 0)
        elif t_idx == -1:
            p_t, p_f = jtfs.phi_f[0], jtfs.sc_freq.psi1_f_fr_up[_f_idx]
            psi_txt = r"$\Psi_{%s, %s, %s}$" % ("-\infty", _f_idx, 0)
        elif f_idx == -1:
            p_t, p_f = jtfs.psi2_f[t_idx], jtfs.sc_freq.phi_f_fr[0]
            psi_txt = r"$\Psi_{%s, %s, %s}$" % (t_idx, "-\infty", 0)

        if t_idx == -1 or f_idx == -1:
            title = (psi_txt, dict(fontsize=17, y=.75))
            p_t, p_f = p_t[0].squeeze(), p_f[0].squeeze()
            Psi = to_time(p_f)[:, None] * to_time(p_t)[None]
            if max_t_bound is not None or zoomed:
                Psi = process_Psi(Psi, max_t_bound, max_f_bound)
            return Psi, title

        # else get spinned wavelets ##########################################
        psi_spin = (jtfs.sc_freq.psi1_f_fr_up if s_idx == 0 else
                    jtfs.sc_freq.psi1_f_fr_down)
        psi_f = psi_spin[_f_idx][0].squeeze()
        psi_t = jtfs.psi2_f[t_idx][0].squeeze()

        f_width = compute_temporal_support(psi_f[None])
        t_width = compute_temporal_support(psi_t[None])
        f_bound = int(2**np.floor(np.log2(f_width)))
        t_bound = int(2**np.floor(np.log2(t_width)))

        # to time
        psi_f = to_time(psi_f)
        psi_t = to_time(psi_t)

        # compute joint wavelet in time
        Psi = psi_f[:, None] * psi_t[None]
        # title
        spin = '+1' if s_idx == 0 else '-1'
        psi_txt = r"$\Psi_{%s, %s, %s}$" % (t_idx, _f_idx, spin)
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
        if not labels:
            title=None
        imshow(Psi, title=title, show=0, ax=ax, ticks=0, borders=borders,
               cmap=cmap)

    # get spinned wavelet arrays & metadata ##################################
    n_rows, n_cols = len(jtfs.psi2_f), len(jtfs.sc_freq.psi1_f_fr_up)
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
    fig, axes = plt.subplots(n_rows * 2 + 1, n_cols + 1, figsize=(8*w, 21*h))
    _txt = "(%s%s" % (part, " part" if part in ('real', 'imag') else "")
    _txt += ", zoomed)" if zoomed else ")"
    if suptitle_y is not None:
        plt.suptitle("Joint wavelet filterbank " + _txt,
                     y=suptitle_y, weight='bold', fontsize=17)

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
                ax = axes[row_idx][f_idx + 1]
                _show(Psi, title=title, ax=ax)

        # (psi_t * psi_f)
        for t_idx in range(n_rows):
            for f_idx in range(n_cols):
                psi_t_idx = t_idx if s_idx == 0 else (n_rows - 1 - t_idx)
                Psi, title, m = imshow_data[(s_idx, psi_t_idx, f_idx)]
                Psi = process_Psi(Psi, max_t_bound, max_f_bound, m)

                row_idx = t_idx + (n_rows + 1) * s_idx
                ax = axes[row_idx][f_idx + 1]
                _show(Psi, title=title, ax=ax)

    # (psi_t * phi_f)
    for t_idx in range(n_rows):
        Psi, title = get_imshow_data(jtfs, t_idx, -1, 0, *bounds)
        if zoomed:
            Psi = process_Psi(Psi, None, None,
                              imshow_data[(0, t_idx, 0)][2])
        ax = axes[t_idx][0]
        _show(Psi, title=title, ax=ax)

    # (phi_t * phi_f)
    Psi, title = get_imshow_data(jtfs, -1, -1, 0, *bounds)
    if zoomed:
        Psi = process_Psi(Psi, None, None,
                          imshow_data[(0, 0, 0)][2])
    ax = axes[n_rows][0]
    _show(Psi, title=title, ax=ax)

    # strip borders of remainders
    for t_idx in range(n_rows + 1, 2*n_rows + 1):
        ax = axes[t_idx][0]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    # tight
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)


def gif_jtfs(Scx, meta, savedir='', base_name='jtfs2d', images_ext='.png',
             overwrite=False, save_images=None, show=None, cmap='jet', norms=None,
             skip_spins=False, skip_unspinned=False, sample_idx=0, inf_token=-1,
             verbose=False, gif_kw=None):
    """Slice heatmaps of JTFS outputs.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    savedir : str
        Path of directory to save GIF/images to. Defaults to current
        working directory.

    base_name : str
        Will save gif with this name, and images with same name enumerated.

    images_ext : str
        Generates images with this format. '.png' (default) is lossless but takes
        more space, '.jpg' is compressed.

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    save_images : bool (default False)
        Whether to save images. Images are always saved if `savepath` is not None,
        but deleted after if `save_images=False`.
        If `True` and `savepath` is None, will save images to current working
        directory (but not gif).

    show : None / bool
        Whether to display images to console. If `savepath` is None, defaults
        to True.

    norms: None / tuple
        Plot color norms for 1) `psi_t * psi_f`, 2) `psi_t * phi_f`, and
        3) `phi_t * psi_f` pairs, respectively.
        Tuple of three (upper limits only, lower assumed 0).
        If None, will norm to `.5 * max(coeffs)`, where coeffs = all joint
        coeffs except `phi_t * phi_f`.

    skip_spins: bool (default False)
        Whether to skip `psi_t * psi_f` pairs.

    skip_unspinned: bool (default False)
        Whether to skip `phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`
        pairs.

    sample_idx : int (default 0)
        Index of sample in batched input to visualize.

    inf_token: int / np.nan
        Placeholder used in `meta` to denote infinity.

    verbose : bool (default False)
        Whether to print to console the location of save file upon success.

    gif_kw : dict / None
        Passed as kwargs to `kymatio.visuals.make_gif`.

    Example
    -------
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(J, N, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        meta = jtfs.meta()

        gif_jtfs(Scx, meta)
    """
    def _title(meta_idx, pair, spin):
        txt = r"$|\Psi_{%s, %s, %s} \star \mathrm{U1}|$"
        values = ns[pair][meta_idx[0]]
        assert values.ndim == 1, values
        mu, l, _ = [int(n) if (float(n).is_integer() and n >= 0) else '-\infty'
                    for n in values]
        return (txt % (mu, l, spin), {'fontsize': 20})

    def _n_n1s(pair):
        n2, n1_fr, _ = ns[pair][meta_idx[0]]
        return np.sum(np.all(ns[pair][:, :2] == np.array([n2, n1_fr]), axis=1))

    def _get_coef(i, pair, meta_idx):
        n_n1s = _n_n1s(pair)
        start, end = meta_idx[0], meta_idx[0] + n_n1s
        if out_list:
            coef = Scx[pair][i]['coef']
        elif out_3D:
            coef = Scx[pair][i]
        else:
            coef = Scx[pair][start:end]
        assert len(coef) == n_n1s
        return coef

    def _save_image():
        path = os.path.join(savedir, f'{base_name}{img_idx[0]}{images_ext}')
        if os.path.isfile(path) and overwrite:
            os.unlink(path)
        if not os.path.isfile(path):
            plt.savefig(path, bbox_inches='tight')
        img_paths.append(path)
        img_idx[0] += 1

    def _viz_spins(Scx, i, norm):
        kup = 'psi_t * psi_f_up'
        kdn = 'psi_t * psi_f_down'
        sup = _get_coef(i, kup, meta_idx)
        sdn = _get_coef(i, kdn, meta_idx)

        _, axes = plt.subplots(1, 2, figsize=(14, 7))
        kw = dict(abs=1, ticks=0, show=0, norm=norm)

        imshow(sup, ax=axes[0], **kw, title=_title(meta_idx, kup, '+1'))
        imshow(sdn, ax=axes[1], **kw, title=_title(meta_idx, kdn, '-1'))
        plt.subplots_adjust(wspace=0.01)
        if save_images or do_gif:
            _save_image()
        if show:
            plt.show()
        plt.close()

        meta_idx[0] += len(sup)

    def _viz_simple(Scx, pair, i, norm):
        coef = _get_coef(i, pair, meta_idx)

        _kw = dict(abs=1, ticks=0, show=0, norm=norm, w=14/12, h=7/12,
                   title=_title(meta_idx, pair, '0'))
        if do_gif:
            # make spacing consistent with up & down
            _, axes = plt.subplots(1, 2, figsize=(14, 7))
            imshow(coef, ax=axes[0], **_kw)
            plt.subplots_adjust(wspace=0.01)
            axes[1].set_frame_on(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        else:
            # optimize spacing for single image
            imshow(coef, **_kw)
        if save_images or do_gif:
            _save_image()
        if show:
            plt.show()
        plt.close()

        meta_idx[0] += len(coef)

    # handle args
    savedir, images_ext, save_images, show, do_gif = _handle_gif_args(
        savedir, base_name, images_ext, save_images, show)

    # set params
    out_3D = bool(meta['n']['psi_t * phi_f'].ndim == 3)
    out_list = isinstance(Scx['S0'], list)
    ns = {pair: meta['n'][pair].reshape(-1, 3) for pair in meta['n']}

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)

    if isinstance(norms, (list, tuple)):
        norms = [(0, n) for n in norms]
    elif isinstance(norms, float):
        norms = [(0, norms) for _ in range(3)]
    else:
        # set to .5 times the max of any joint coefficient (except phi_t * phi_f)
        mx = np.max([(c['coef'] if out_list else c).max()
                     for pair in Scx for c in Scx[pair]
                     if pair not in ('S0', 'S1', 'phi_t * phi_f')])
        norms = [(0, .5 * mx)] * 5

    # spinned pairs ##########################################################
    img_paths = []
    img_idx = [0]
    meta_idx = [0]
    if not skip_spins:
        i = 0
        while True:
            _viz_spins(Scx, i, norms[0])
            i += 1
            if meta_idx[0] > len(ns['psi_t * psi_f_up']) - 1:
                break

    # unspinned pairs ########################################################
    if not skip_unspinned:
        pairs = ('psi_t * phi_f', 'phi_t * psi_f', 'phi_t * phi_f')
        for j, pair in enumerate(pairs):
            meta_idx = [0]
            i = 0
            while True:
                _viz_simple(Scx, pair, i, norms[1 + j])
                i += 1
                if meta_idx[0] > len(ns[pair]) - 1:
                    break

    # make gif & cleanup #####################################################
    if do_gif:
        if gif_kw is None:
            gif_kw = {}
        savepath = os.path.join(savedir, base_name + '.gif')
        make_gif(loaddir=savedir, savepath=savepath, ext=images_ext,
                 overwrite=overwrite, delimiter=base_name, verbose=verbose,
                 **gif_kw)
    if not save_images:
        for path in img_paths:
            if os.path.isfile(path):
                os.unlink(path)


def gif_jtfs_3D(packed, savedir='', base_name='jtfs3d', images_ext='.png',
                cmap='turbo', cmap_norm=.5, axes_labels=('xi2', 'xi1_fr', 'xi1'),
                overwrite=False, save_images=False,
                width=800, height=800, surface_count=30, opacity=.2, zoom=1.61,
                verbose=True, gif_kw=None):
    """Generate and save GIF of 3D JTFS slices.

    Parameters
    ----------
    packed : tensor, 4D
        Output of `kymatio.toolkit.pack_coeffs_jtfs`.

    savedir, base_name, images_ext, overwrite :
        See `help(kymatio.visuals.gif_jtfs)`.

    cmap : str
        Colormap to use.

    cmap_norm : float
        Colormap norm to use, as fraction of maximum value of `packed`
        (i.e. `norm=(0, cmap_norm * packed.max())`).

    axes_labels : tuple[str]
        Names of last three dimensions of `packed`. E.g. `structure==2`
        will output `(n2, n1_fr, n1, t)`.

    width : int
        2D width of each image (GIF frame).

    height : int
        2D height of each image (GIF frame).

    surface_count : int
        Greater improves 3D detail of each frame, but takes longer to render.

    opacity : float
        Lesser makes 3D surfaces more transparent, exposing more detail.

    zoom : float
        Zoom factor on each 3D frame.

    verbose : bool (default True)
        Whether to print GIF generation progress.

    gif_kw : dict / None
        Passed as kwargs to `kymatio.visuals.make_gif`.

    Example
    -------
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(J, N, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        meta = jtfs.meta()

        packed = toolkit.pack_coeffs_jtfs(Scx, meta, structure=2,
                                          separate_lowpass=True)
        packed_spinned = packed[0]
        packed_spinned = packed_spinned.transpose(-1, 0, 1, 2)  # time first
        gif_jtfs_3D(packed_spinned, savedir='')
    """
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        print("\n`plotly.graph_objs` is needed for `gif_jtfs_3D`.")
        raise e

    # handle args
    savedir, images_ext, save_images, *_ = _handle_gif_args(
        savedir, base_name, images_ext, save_images, show=False)

    # handle labels
    supported = ('t', 'xi2', 'xi1_fr', 'xi1')
    for label in axes_labels:
        if label not in supported:
            raise ValueError(("unsupported `axes_labels` element: {} -- must "
                              "be one of: {}").format(
                                  label, ', '.join(supported)))
    frame_label = [label for label in supported if label not in axes_labels][0]

    # 3D meshgrid
    def slc(i, g):
        label = axes_labels[i]
        start = {'xi1': 0,  'xi2': 0,  't': 0, 'xi1_fr': -.5}[label]
        end   = {'xi1': .5, 'xi2': .5, 't': 1, 'xi1_fr':  .5}[label]
        return slice(start, end, g*1j)

    a, b, c = packed.shape[1:]
    X, Y, Z = np.mgrid[slc(0, a), slc(1, b), slc(2, c)]
    if 'xi1_fr' in axes_labels:
        # distinguish + and - spin
        idx = axes_labels.index('xi1_fr')
        # flip idx-th dim to align with meshgrid
        _slc = (slice(None),) * (1 + idx) + (slice(None, None, -1),)
        packed = packed[_slc]

    # camera focus, colormap norm
    eye = np.array([2.5, .3, 2]) / zoom
    mx = cmap_norm * packed.max()

    # gif configs
    volume_kw = dict(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        colorscale=cmap,
        showscale=False,
        cmin=0,
        cmax=mx,
    )
    layout_kw = dict(
        margin_pad=0,
        margin_l=0,
        margin_r=0,
        margin_t=0,
        title_pad_t=0,
        title_pad_b=0,
        margin_autoexpand=False,
        scene_aspectmode='cube',
        width=width,
        height=height,
        scene=dict(
            xaxis_title=axes_labels[0],
            yaxis_title=axes_labels[1],
            zaxis_title=axes_labels[2],
        ),
        scene_camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        ),
    )

    # generate gif frames ####################################################
    img_paths = []
    for k, vol4 in enumerate(packed):
        fig = go.Figure(go.Volume(value=vol4.flatten(), **volume_kw))
        fig.update_layout(
            **layout_kw,
            title={'text': f"{frame_label}={k}",
                   'x':.5, 'y':.09,
                   'xanchor': 'center', 'yanchor': 'top'}
        )

        savepath = os.path.join(savedir, f'{base_name}{k}{images_ext}')
        if os.path.isfile(savepath) and overwrite:
            os.unlink(savepath)
        fig.write_image(savepath)
        img_paths.append(savepath)
        if verbose:
            print("{}/{} frames done".format(k + 1, len(packed)), flush=True)

    # make gif ###############################################################
    try:
        if gif_kw is None:
            gif_kw = {}
        savepath = os.path.join(savedir, f'{base_name}.gif')
        make_gif(loaddir=savedir, savepath=savepath, ext=images_ext,
                 delimiter=base_name, overwrite=overwrite, verbose=verbose,
                 **gif_kw)
    finally:
        if not save_images:
            # delete images
            for path in img_paths:
                if os.path.isfile(path):
                    os.unlink(path)


def energy_profile_jtfs(Scx, meta, x=None, pairs=None, kind='l2', plots=True,
                        **plot_kw):
    """Plot & print relevant energy information across coefficient pairs.
    Works for all `'dict' in out_type` and `out_exclude`.
    Also see `help(kymatio.toolkit.coeff_energy)`.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    x : tensor, optional
        Original input to print `E_out / E_in`.

    pairs: None / list/tuple[str]
        Computes energies for these pairs in provided order. None will compute
        for all in default order:
            ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_down')

    kind : str['l1', 'l2']
        - L1: `sum(abs(x))`
        - L2: `sum(abs(x)**2)` -- actually L2^2

    Returns
    -------
    energies: list[float]
        List of coefficient energies.
    pair_energies: dict[str: float]
        Keys are pairs, values are sums of all pair's coefficient energies.
    """
    if not isinstance(Scx, dict):
        raise NotImplementedError("input must be dict. Set out_type='dict:array' "
                                  "or 'dict:list'.")
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target="L1 norm" if kind == 'l1' else "Energy")
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_energy(
        Scx, meta, pair, aggregate=False, kind=kind)

    # compute, plot, print
    energies, pair_energies = _iterate_coeff_pairs(
        Scx, meta, fn, pairs, plots=plots, titles=titles, **plot_kw)

    # E_out / E_in
    if x is not None:
        e_total = np.sum(energies)
        print("E_out / E_in = %.3f" % (e_total / energy(x)))
    return energies, pair_energies


def coeff_distance_jtfs(Scx0, Scx1, meta0, meta1=None, pairs=None, kind='l2',
                        plots=True, **plot_kw):
    # TODO doc
    if not all(isinstance(Scx, dict) for Scx in (Scx0, Scx1)):
        raise NotImplementedError("inputs must be dict. Set "
                                  "out_type='dict:array' or 'dict:list'.")
    if meta1 is None:
        meta1 = meta0

    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target="L1 norm" if kind == 'l1' else "Energy")
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_distance(*Scx, *meta, pair, kind=kind)

    # compute, plot, print
    # TODO instead return as "pairs:flat", "pairs:slices"?
    distances, pair_distances = _iterate_coeff_pairs(
        (Scx0, Scx1), (meta0, meta1), fn, pairs, plots=plots,
        titles=titles, **plot_kw)

    return distances, pair_distances


def _iterate_coeff_pairs(Scx, meta, fn, pairs=None, plots=True, titles=None,
                         **plot_kw):
    # in case multiple meta passed
    meta0 = meta[0] if isinstance(meta, tuple) else meta
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)

    # extract energy info
    energies = []
    pair_energies = {}
    idxs = [0]
    for pair in compute_pairs:
        if pair not in meta0['n']:
            continue
        E_flat, E_slices = fn(Scx, meta, pair)
        pair_energies[pair] = E_slices[::-1]
        energies.extend(E_slices[::-1])
        # don't repeat 0
        idxs.append(len(energies) - 1 if len(energies) != 1 else 1)

    # format & plot ##########################################################
    energies = np.array(energies)
    ticks = np.arange(len(energies))
    vlines = (idxs, {'color': 'tab:red', 'linewidth': 1})

    if titles is None:
        titles = ['', '']
    if plots:
        scat(ticks[idxs], energies[idxs], s=20)
        plot_kw['ylims'] = plot_kw.get('ylims', (0, None))
        plot(energies, vlines=vlines, title=titles[0], show=1, **plot_kw)

    # cumulative sum
    energies_cs = np.cumsum(energies)

    if plots:
        scat(ticks[idxs], energies_cs[idxs], s=20)
        plot(energies_cs, vlines=vlines, title=titles[1], show=1, **plot_kw)

    # print report ###########################################################
    def sig_figs(x, n_sig=3):
        s = str(x)
        nondecimals = len(s.split('.')[0]) - int(s[0] == '0')
        decimals = max(n_sig - nondecimals, 0)
        return s.lstrip('0')[:decimals + nondecimals + 1].rstrip('.')

    e_total = np.sum(energies)
    pair_energies_sum = {pair: np.sum(pair_energies[pair])
                         for pair in pair_energies}
    nums = [sig_figs(e, n_sig=3) for e in pair_energies_sum.values()]
    longest_num = max(map(len, nums))

    if plots:
        i = 0
        for pair in compute_pairs:
            E_pair = pair_energies_sum[pair]
            e_perc = sig_figs(E_pair / e_total * 100, n_sig=3)
            print("{} ({}%) -- {}".format(
                nums[i].ljust(longest_num), str(e_perc).rjust(4), pair))
            i += 1
    return energies, pair_energies


def compare_distances_jtfs(pair_distances, pair_distances_ref, plots=True,
                           verbose=True, title=None):
    # don't modify external
    pd0, pd1 = deepcopy(pair_distances), deepcopy(pair_distances_ref)

    ratios, stats = {}, {}
    for pair in pd0:
        p0, p1 = np.asarray(pd0[pair]), np.asarray(pd1[pair])
        # threshold out small points
        idxs = np.where((p0 < .001*p0.max()).astype(int) +
                        (p1 < .001*p1.max()).astype(int))[0]
        p0[idxs], p1[idxs] = 1, 1
        R = p0 / p1
        ratios[pair] = R
        stats[pair] = dict(mean=R.mean(), min=R.min(), max=R.max())

    if plots:
        if title is None:
            title = ''
        _title = _make_titles_jtfs(list(ratios), f"Distance ratios | {title}")[0]
        vidxs = np.cumsum([len(r) for r in ratios.values()])
        ratios_flat = np.array([r for rs in ratios.values() for r in rs])
        plot(ratios_flat, ylims=(0, None), title=_title,
             hlines=(1,     dict(color='tab:red', linestyle='--')),
             vlines=(vidxs, dict(color='k', linewidth=1)))
        scat(idxs, ratios_flat[idxs], color='tab:red', show=1)
    if verbose:
        print("mean  min   max   | pair")
        for pair in ratios:
            print("{:<5.2f} {:<5.2f} {:<5.2f} | {}".format(
                *list(stats[pair].values()), pair))
    return ratios, stats


def make_gif(loaddir, savepath, duration=250, start_end_pause=3, ext='.png',
             delimiter='', overwrite=False, HD=None, verbose=False):
    """Makes gif out of images in `loaddir` directory with `ext` extension,
    and saves to `savepath`.

    Parameters
    ----------
    loaddir : str
        Path to directory from which to fetch images to use as GIF frames.

    savepath : path
        Save path, must end with '.gif'.

    duration : int
        Interval between each GIF frame, in milliseconds.

    start_end_pause : int / tuple[int]
        Number of times to repeat the start and end frames, which multiplies
        their `duration`; if tuple, first element is for start, second for end.

    ext : str
        Images filename extension.

    delimiter : str
        Substring common to all iamge filenames, e.g. 'img' for 'img0.png',
        'img1.png', ... .

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    HD : bool / None
        If True, will preserve image quality in GIFs and use `imageio`.
        Defaults to True if `imageio` is installed, else falls back on
        `PIL.Image`.

    verbose : bool (default False)
        Whether to print to console the location of save file upon success.
    """
    # handle `HD`
    if HD or HD is None:
        try:
            import imageio
            HD = True
        except ImportError as e:
            if HD:
                print("`HD=True` requires `imageio` installed")
                raise e
            else:
                try:
                    from PIL import Image
                except ImportError as e:
                    print("`make_gif` requires `imageio` or `PIL` installed.")
                    raise e
                HD = False

    # fetch frames
    loaddir = os.path.abspath(loaddir)
    names = [n for n in os.listdir(loaddir)
             if (n.startswith(delimiter) and n.endswith(ext))]
    names = sorted(names, key=lambda p: int(
        ''.join(s for s in p.split(os.sep)[-1] if s.isdigit())))
    paths = [os.path.join(loaddir, n) for n in names]
    frames = [(imageio.imread(p) if HD else Image.open(p))
              for p in paths]

    # handle frame duplication to increase their duration
    if start_end_pause is not None:
        if not isinstance(start_end_pause, (tuple, list)):
            start_end_pause = (start_end_pause, start_end_pause)
        for repeat_start in range(start_end_pause[0]):
            frames.insert(0, frames[0])
        for repeat_end in range(start_end_pause[1]):
            frames.append(frames[-1])

    if os.path.isfile(savepath) and overwrite:
        # delete if exists
        os.unlink(savepath)
    # save
    if HD:
        imageio.mimsave(savepath, frames, fps=1000/duration)
    else:
        frame_one = frames[0]
        frame_one.save(savepath, format="GIF", append_images=frames,
                       save_all=True, duration=duration, loop=0)
    if verbose:
        print("Saved gif to", savepath)


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
         ax=None, fig=None, squeeze=True, auto_xlims=None, **kw):
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

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        y = y if isinstance(y, list) or not squeeze else y.squeeze()
        x = np.arange(len(y))
    elif y is None:
        x = x if isinstance(x, list) or not squeeze else x.squeeze()
        y = x
        x = np.arange(len(x))
    x = x if isinstance(x, list) or not squeeze else x.squeeze()
    y = y if isinstance(y, list) or not squeeze else y.squeeze()

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

    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks)

    if title is not None:
        _title(title, ax=ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def scat(x, y=None, title=None, show=0, s=18, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None, ticks=1,
         complex=False, abs=False, xlabel=None, ylabel=None, ax=None, fig=None,
         auto_xlims=None, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

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
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def plotscat(*args, **kw):
    show = kw.pop('show', False)
    plot(*args, **kw)
    scat(*args, **kw)
    if show:
        plt.show()


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

#### misc / utils ############################################################
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
                xlims=None, ylims=None, xlabel=None, ylabel=None,
                auto_xlims=True):
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)

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


def _get_compute_pairs(pairs, meta):
    # enforce pair order
    if pairs is None:
        pairs_all = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                     'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_down')
    else:
        pairs_all = pairs if not isinstance(pairs, str) else [pairs]
    compute_pairs = []
    for pair in pairs_all:
        if pair in meta['n']:
            compute_pairs.append(pair)
    return compute_pairs


def _make_titles_jtfs(compute_pairs, target):
    """For energies and distances."""
    # make `titles`
    titles = []
    pair_aliases = {'psi_t * phi_f': '* phi_f', 'phi_t * psi_f': 'phi_t *',
                    'psi_t * psi_f_up': 'up', 'psi_t * psi_f_down': 'down'}
    title = "%s | " % target
    for pair in compute_pairs:
        if pair in pair_aliases:
            title += "{}, ".format(pair_aliases[pair])
        else:
            title += "{}, ".format(pair)
    title = title.rstrip(', ')
    titles.append(title)

    title = "cumsum(%s)" % target
    titles.append(title)
    return titles


def _handle_gif_args(savedir, base_name, images_ext, save_images, show):
    do_gif = bool(savedir is not None)
    if save_images is None:
        if savedir is None:
            save_images = bool(not show)
        else:
            save_images = False
    if show is None:
        show = bool(not save_images and not do_gif)

    if savedir is None and save_images:
        savedir = ''
    if savedir is not None:
        savedir = os.path.abspath(savedir)

    if not images_ext.startswith('.'):
        images_ext = '.' + images_ext

    return savedir, images_ext, save_images, show, do_gif
