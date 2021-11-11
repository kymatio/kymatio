# -*- coding: utf-8 -*-
"""Joint Time-Frequency Scattering related tests."""
import pytest
import numpy as np
import warnings
from pathlib import Path
from copy import deepcopy

from kymatio import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import (drop_batch_dim_jtfs, jtfs_to_numpy, coeff_energy,
                             fdts, echirp, coeff_energy_ratios, energy,
                             est_energy_conservation, rel_l2, rel_ae,
                             validate_filterbank_tm, validate_filterbank_fr,
                             pack_coeffs_jtfs, tensor_padded, normalize)
from kymatio.visuals import (coeff_distance_jtfs, compare_distances_jtfs,
                             energy_profile_jtfs)
from kymatio.scattering1d.filter_bank import compute_temporal_width, gauss_1d
from utils import cant_import

# backend to use for all tests (except `test_backends`)
# NOTE: non-'numpy' skips `test_meta()` and `test_lp_sum()`
default_backend = ('numpy', 'torch', 'tensorflow')[0]
# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to print assertion errors rather than raising them in `test_output()`
output_test_print_mode = 1
# set True to print assertion values of certain tests
metric_verbose = 1

# used to load saved coefficient outputs
test_data_dir = Path(__file__).parent


def test_alignment():
    """Ensure A.M. cosine's peaks are aligned across joint slices."""
    N = 1025
    J = 7
    Q = 16
    Q_fr = 2
    F = 4

    # generate A.M. cosine ###################################################
    f1, f2 = 8, 256
    t = np.linspace(0, 1, N, 1)
    a = (np.cos(2*np.pi * f1 * t) + 1) / 2
    c = np.cos(2*np.pi * f2 * t)
    x = a * c

    # scatter ################################################################
    for out_3D in (True, False):
      for sampling_psi_fr in ('resample', 'exclude'):
        if sampling_psi_fr == 'exclude' and out_3D:
            continue  # incompatible
        for J_fr in (3, 5):
          out_type = ('dict:array' if out_3D else
                      'dict:list')  # for convenience
          test_params = dict(out_3D=out_3D,
                             sampling_filters_fr=(sampling_psi_fr, 'resample'))
          test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                      test_params.items())
          jtfs = TimeFrequencyScattering1D(
              J, N, Q, J_fr=J_fr, Q_fr=Q_fr, F=F, average=True, average_fr=True,
              aligned=True, out_type=out_type, frontend=default_backend,
              pad_mode='zero', pad_mode_fr='zero', **test_params)

          Scx = jtfs(x)
          Scx = drop_batch_dim_jtfs(Scx)
          Scx = jtfs_to_numpy(Scx)

          # assert peaks share an index #################################
          def max_row_idx(c):
              coef = c['coef'] if 'list' in out_type else c
              return np.argmax(np.sum(coef**2, axis=-1))

          first_coef = Scx['psi_t * psi_f_up'][0]
          mx_idx = max_row_idx(first_coef)
          for pair in Scx:
              if pair in ('S0', 'S1'):  # joint only
                  continue

              for i, c in enumerate(Scx[pair]):
                  mx_idx_i = max_row_idx(c)
                  assert abs(mx_idx_i - mx_idx) < 2, (
                      "{} != {} -- Scx[{}][{}]\n{}").format(
                          mx_idx_i, mx_idx, pair, i, test_params_str)

          if J_fr == 3:
              # assert not all J_pad_frs are same so test covers this case
              assert_pad_difference(jtfs, test_params_str)


def test_shapes():
    """Ensure `out_3D=True` joint coeff slices have same shape."""
    N = 1024
    J = 6
    Q = 16

    x = np.random.randn(N)

    # scatter ################################################################
    for oversampling in (0, 1):
      for oversampling_fr in (0, 1):
        for aligned in (True, False):
          test_params = dict(oversampling=oversampling,
                             oversampling_fr=oversampling_fr, aligned=aligned)
          test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                      test_params.items())

          jtfs = TimeFrequencyScattering1D(
              J, N, Q, J_fr=4, Q_fr=2, average=True, average_fr=True,
              out_type='dict:array', out_3D=True, aligned=aligned,
              oversampling=oversampling, oversampling_fr=oversampling_fr,
              frontend=default_backend)
          try:
              _ = jtfs(x)  # shapes must equal for this not to error
          except Exception as e:
              print(test_params_str)
              raise e


def test_jtfs_vs_ts():
    """Test JTFS sensitivity to FDTS (frequency-dependent time shifts), and that
    time scattering is insensitive to it.
    """
    # design signal
    N = 2048
    f0 = N // 20
    n_partials = 5
    partials_f_sep = 1.6
    total_shift = N//14
    seg_len = N//6

    x, xs = fdts(N, n_partials, total_shift, f0, seg_len,
                 partials_f_sep=partials_f_sep)

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = (8, 2)
    kw = dict(Q=Q, J=J, shape=N, max_pad_factor=1, frontend=default_backend)
    ts = Scattering1D(pad_mode="zero", out_type='array', **kw)
    jtfs = TimeFrequencyScattering1D(Q_fr=2, J_fr=4, average_fr=True,
                                     out_3D=True, out_type='dict:array', **kw,
                                     sampling_filters_fr=('resample', 'resample'))

    # scatter
    ts_x  = ts(x)
    ts_xs = ts(xs)

    jtfs_x_all  = jtfs(x)
    jtfs_xs_all = jtfs(xs)
    jtfs_x_all  = jtfs_to_numpy(jtfs_x_all)
    jtfs_xs_all = jtfs_to_numpy(jtfs_xs_all)
    jtfs_x  = concat_joint(jtfs_x_all)
    jtfs_xs = concat_joint(jtfs_xs_all)  # compare against joint coeffs only

    l2_ts   = float(rel_l2(ts_x, ts_xs))
    l2_jtfs = float(rel_l2(jtfs_x, jtfs_xs))

    # max ratio limited by `N`; can do better with longer input
    # and by comparing only against up & down, and via per-coeff basis
    assert l2_jtfs / l2_ts > 25, ("\nJTFS/TS: %s \nTS: %s\nJTFS: %s"
                                  ) % (l2_jtfs / l2_ts, l2_ts, l2_jtfs)
    assert l2_ts < .008, "TS: %s" % l2_ts

    if metric_verbose:
        print(("\nFDTS sensitivity:\n"
               "JTFS/TS = {:.1f}\n"
               "TS      = {:.4f}\n").format(l2_jtfs / l2_ts, l2_ts))


def test_freq_tp_invar():
    """Test frequency transposition invariance."""
    # design signal
    N = 2048
    f0 = N // 12
    f1 = f0 / np.sqrt(2)
    n_partials = 5
    seg_len = N//8

    x0 = fdts(N, n_partials, f0=f0, seg_len=seg_len)[0]
    x1 = fdts(N, n_partials, f0=f1, seg_len=seg_len)[0]

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    J_fr = 5
    F_all = [32, 64]

    pair_distances, global_distances = [], []
    for F in F_all:
        jtfs = TimeFrequencyScattering1D(J=J, Q=16, Q_fr=1, J_fr=J_fr, shape=N,
                                         F=F, average_fr=True, out_3D=False,
                                         out_type='dict:array',
                                         out_exclude=('S0', 'S1'),
                                         # pad_mode='zero', pad_mode_fr='zero',
                                         pad_mode='reflect',
                                         pad_mode_fr='conj-reflect-zero',
                                         sampling_filters_fr=(
                                             'resample', 'resample'),
                                         frontend=default_backend)
        # scatter
        jtfs_x0_all = jtfs(x0)
        jtfs_x1_all = jtfs(x1)
        jtfs_x0_all = jtfs_to_numpy(jtfs_x0_all)
        jtfs_x1_all = jtfs_to_numpy(jtfs_x1_all)

        # compute & append distances
        _, pair_dist = coeff_distance_jtfs(jtfs_x0_all, jtfs_x1_all,
                                           jtfs.meta(), plots=False)
        pair_distances.append(pair_dist)

        jtfs_x0 = concat_joint(jtfs_x0_all)
        jtfs_x1 = concat_joint(jtfs_x1_all)
        global_distances.append(float(rel_l2(jtfs_x0, jtfs_x1)))

    if metric_verbose:
        print("\nFrequency transposition invariance stats:")

    # compute stats & assert
    _, stats = compare_distances_jtfs(*pair_distances, plots=0,
                                      verbose=metric_verbose, title="F: 32 vs 64")
    maxs, means = zip(*[(s['max'], s['mean']) for s in stats.values()])
    max_max, mean_mean = max(maxs), np.mean(means)
    # best case must attain at least twice the invariance
    assert max_max > 2, max_max
    # global mean ratio should exceed unity
    assert mean_mean > 1.4, mean_mean

    if metric_verbose:
        print("max_max, mean_mean = {:.2f}, {:.2f}".format(max_max, mean_mean))
        print("Global L2: (F=32, F=64, ratio) = ({:.3f}, {:.3f}, {:.3f})".format(
            *global_distances, global_distances[0] / global_distances[1]))


def test_up_vs_down():
    """Test that echirp yields significant disparity in up vs down coeffs."""
    N = 2048
    x = echirp(N, fmin=64)

    if metric_verbose:
        print("\nFDTS directional sensitivity; E_down / E_up:")

    m_th = (170, 420)
    l2_th = (85, 470)
    for i, pad_mode in enumerate(['reflect', 'zero']):
        pad_mode_fr = 'conj-reflect-zero' if pad_mode == 'reflect' else 'zero'
        jtfs = TimeFrequencyScattering1D(shape=N, J=8, Q=16, J_fr=4, F=4, Q_fr=2,
                                         average_fr=True, out_type='dict:array',
                                         pad_mode=pad_mode,
                                          sampling_filters_fr=(
                                              'resample', 'resample'),
                                         pad_mode_fr=pad_mode_fr,
                                         frontend=default_backend)
        Scx = jtfs(x)
        Scx = jtfs_to_numpy(Scx)
        jmeta = jtfs.meta()

        r = coeff_energy_ratios(Scx, jmeta)
        r_m = r.mean()

        E_up   = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_up')
        E_down = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_down')
        r_l2 = E_down / E_up

        if metric_verbose:
            print(("Global:     {0:<5.1f} -- '{1}' pad\n"
                   "Slice mean: {2:<5.1f} -- '{1}' pad").format(
                       r_l2, pad_mode, r_m))
        assert r_l2 > l2_th[i], "{} < {} | '{}'".format(r_l2, l2_th[i], pad_mode)
        assert r_m  > m_th[i],  "{} < {} | '{}'".format(r_m,  m_th[i],  pad_mode)


def test_sampling_psi_fr_exclude():
    """Test that outputs of `sampling_psi_fr=='exclude'` are a subset of
    `==True` (i.e. equal wherever both exist).
    """
    N = 1024
    x = echirp(N)

    params = dict(shape=N, J=9, Q=8, J_fr=3, Q_fr=2, F=4, average_fr=True,
                  out_type='dict:list', frontend=default_backend)
    test_params_str = '\n'.join(f'{k}={v}' for k, v in params.items())
    jtfs0 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('resample', 'resample'))
    jtfs1 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('exclude', 'resample'))

    # required otherwise 'exclude' == 'resample'
    assert_pad_difference(jtfs0, test_params_str)
    # reproduce case with different J_pad_fr
    assert jtfs0.J_pad_frs != jtfs1.J_pad_frs, jtfs0.J_pad_frs

    Scx0 = jtfs0(x)
    Scx1 = jtfs1(x)
    Scx0 = jtfs_to_numpy(Scx0)
    Scx1 = jtfs_to_numpy(Scx1)

    # assert equality where `n` metas match
    # if `n` don't match, assert J_pad_fr is below maximum
    for pair in Scx1:
        i1 = 0
        for i0, c in enumerate(Scx1[pair]):
            s0, s1 = Scx0[pair][i0], Scx1[pair][i1]
            n0, n1 = s0['n'], s1['n']
            c0, c1 = s0['coef'], s1['coef']
            info = "{}, (i0, i1)=({}, {}); (n0, n1)=({}, {})".format(
                pair, i0, i1, n0, n1)

            is_joint = bool(pair not in ('S0', 'S1'))
            if is_joint:
                pad, pad_max = jtfs1.J_pad_frs[n0[0]], jtfs1.J_pad_frs_max
            if n0 != n1:
                assert is_joint, (
                    "found mismatch in time scattered coefficients\n%s" % info)
                # Mismatched `n` should only happen for mismatched `pad_fr`.
                # Check 1's pad as indexed by 0, since n0 lags n1 and might
                # have e.g. pad1[n0=5]==(max-1), pad[n1=6]==max, but we're still
                # iterating n==5 so appropriate comparison is at 5
                assert pad != pad_max, (
                    "{} == {} | {}\n(must have sub-maximal `J_pad_fr` for "
                    "mismatched `n`)").format(pad, pad_max, info)
                continue

            assert c0.shape == c1.shape, "shape mismatch: {} != {} | {}".format(
                c0.shape, c1.shape, info)
            ae = rel_ae(c0, c1)
            # due to different energy renorms (LP sum)
            atol = 1e-8 if (not is_joint or pad == pad_max) else 1e-2
            assert np.allclose(c0, c1, atol=atol), (
                "{} | MeanAE={:.2e}, MaxAE={:.2e}").format(
                    info, ae.mean(), ae.max())
            i1 += 1


def test_max_pad_factor_fr():
    """Test that low and variable `max_pad_factor_fr` works."""
    N = 1024
    x = echirp(N)

    for aligned in (True, False)[1:]:
        for sampling_filters_fr in ('resample', 'exclude', 'recalibrate'):
          for max_pad_factor_fr in (0, 1, [2, 1, 0]):
            F = 128 if sampling_filters_fr == 'recalibrate' else 16
            test_params = dict(aligned=aligned,
                               sampling_filters_fr=sampling_filters_fr,
                               max_pad_factor_fr=max_pad_factor_fr)
            test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                        test_params.items())

            try:
                jtfs = TimeFrequencyScattering1D(
                    shape=N, J=9, Q=12, J_fr=4, Q_fr=1, F=F, average_fr=True,
                    out_3D=True, **test_params, frontend=default_backend)
            except Exception as e:
                if not ("same `J_pad_fr`" in str(e) and
                        sampling_filters_fr == 'recalibrate'):
                    print("Failed on %s with" % test_params_str)
                    raise e
                else:
                    continue
            assert_pad_difference(jtfs, test_params_str)

            try:
                _ = jtfs(x)
            except Exception as e:
                print("Failed on %s with" % test_params_str)
                raise e


def test_out_exclude():
    """Test that `out_exclude` works as expected."""
    N = 512
    params = dict(shape=N, J=4, Q=4, J_fr=4, average=False, average_fr=True,
                  out_type='dict:list', frontend=default_backend)
    x = np.random.randn(N)

    all_pairs = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                 'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_down')
    out_excludes = [
        ('S0', 'psi_t * psi_f_up'),
        ('psi_t * psi_f_down', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f'),
        ('S1', 'psi_t * psi_f_up', 'psi_t * psi_f_down'),
    ]
    for out_exclude in out_excludes:
        jtfs = TimeFrequencyScattering1D(**params, out_exclude=out_exclude)
        out = jtfs(x)
        jmeta = jtfs.meta()

        for pair in out:
            assert pair not in out_exclude, pair
            assert pair in all_pairs, pair  # ensure nothing else was inserted

        for field in jmeta:
            for pair in jmeta[field]:
                assert pair not in out_exclude, (field, pair)
                assert pair in all_pairs, (field, pair)

    # ensure invalid pair is caught
    with pytest.raises(ValueError) as record:
        jtfs = TimeFrequencyScattering1D(**params, out_exclude=('banana',))
    assert "invalid coefficient" in record.value.args[0]


def test_global_averaging():
    """Test that `T==N` and `F==pow2(N_frs_max)` doesn't error, and outputs
    close to `T==N-1` and `F==pow2(N_frs_max)-1`
    """
    np.random.seed(0)
    N = 512
    params = dict(shape=N, J=9, Q=4, J_fr=5, Q_fr=2, average=True,
                  average_fr=True, out_type='dict:array', pad_mode='reflect',
                  pad_mode_fr='conj-reflect-zero', max_pad_factor=None,
                  max_pad_factor_fr=None, frontend=default_backend,
                  sampling_filters_fr=('resample', 'resample'))
    x = echirp(N)
    x += np.random.randn(N)

    outs = {}
    metas = {}
    Ts, Fs = (N - 1, N), (2**5 - 1, 2**5)
    for T in Ts:
        # N_frs_max ~= Q*max(p2['j'] for p2 in psi2_f); found 29 at runtime
        for F in Fs:
            jtfs = TimeFrequencyScattering1D(**params, T=T, F=F)
            assert (jtfs.average_fr_global if F == Fs[-1] else
                    not jtfs.average_fr_global)
            assert (jtfs.average_global if T == Ts[-1] else
                    not jtfs.average_global)

            out = jtfs(x)
            out = jtfs_to_numpy(out)
            outs[ (T, F)] = out
            metas[(T, F)] = jtfs.meta()

    T0F0 = coeff_energy(outs[(Ts[0], Fs[0])], metas[(Ts[0], Fs[0])])
    T0F1 = coeff_energy(outs[(Ts[0], Fs[1])], metas[(Ts[0], Fs[1])])
    T1F0 = coeff_energy(outs[(Ts[1], Fs[0])], metas[(Ts[1], Fs[0])])
    T1F1 = coeff_energy(outs[(Ts[1], Fs[1])], metas[(Ts[1], Fs[1])])

    if metric_verbose:
        print("\nGlobal averaging reldiffs:")

    th = .15
    for pair in T0F0:
        ref = T0F0[pair]
        reldiff01 = abs(T0F1[pair] - ref) / ref
        reldiff10 = abs(T1F0[pair] - ref) / ref
        reldiff11 = abs(T1F1[pair] - ref) / ref
        assert reldiff01 < th, "%s > %s | %s" % (reldiff01, th, pair)
        assert reldiff10 < th, "%s > %s | %s" % (reldiff10, th, pair)
        assert reldiff11 < th, "%s > %s | %s" % (reldiff11, th, pair)

        if metric_verbose:
            print("(01, 10, 11) = ({:.2e}, {:.2e}, {:.2e}) | {}".format(
                reldiff01, reldiff10, reldiff11, pair))


def test_lp_sum():
    """Test that filterbank energy renormalization works as expected."""
    def _get_th_and_text(k, j0, th_loc):
        if k is not None:  # temporal
            s = "k=%s" % k
            peak_target = 2
        elif j0 is not None:  # frequential
            s = "j0=%s" % j0
            peak_target = 1
        th = peak_target * th_loc
        return th, peak_target, s

    def check_above(lp, test_params_str, k=None, j0=None):
        th, peak_target, s = _get_th_and_text(k, j0, th_above)
        assert lp.max() - peak_target < th, (
            "{} - {} > {} | {}\n{}").format(lp.max(), peak_target, th, s,
                                            test_params_str)

    def check_below(lp, test_params_str, psi_fs, k=None, j0=None):
        th, peak_target, s = _get_th_and_text(k, j0, th_below)
        first_peak = np.argmax(psi_fs[-1])
        last_peak  = np.argmax(psi_fs[0])
        lp_min = lp[first_peak:last_peak + 1].min()

        assert peak_target - lp_min > th, (
            "{} - {} < {} | between peaks {} and {} | {}\n{}"
            ).format(peak_target, lp_min, th, first_peak, last_peak,
                     s, test_params_str)

    if default_backend != 'numpy':
        # filters don't change
        warnings.warn("`test_lp_sum()` skipped per non-'numpy' `default_backend`")
        return

    N = 1024
    J = int(np.log2(N))
    common_params = dict(shape=N, J=J, Q_fr=3, frontend=default_backend)
    th_above = 1.5e-2
    th_below = .5

    for Q in (1, 8, 16):
      for r_psi in (np.sqrt(.5), .85):
        for max_pad_factor in (None, 1):
          for max_pad_factor_fr in (None, 1):
            for sampling_filters_fr in ('resample', 'exclude', 'recalibrate'):
              test_params = dict(Q=Q, r_psi=r_psi, max_pad_factor=max_pad_factor,
                                 max_pad_factor_fr=max_pad_factor_fr,
                                 sampling_filters_fr=sampling_filters_fr,
                                 # ensure total LP sum is poorly behaved
                                 # (but does not affect psi-only LP sum)
                                 T=2**(common_params['J'] - 1))
              test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                          test_params.items())
              try:
                  jtfs = TimeFrequencyScattering1D(**common_params, **test_params)
              except Exception as e:
                  print(test_params_str)
                  raise e

              # temporal filterbank
              for order, psi_fs in enumerate([jtfs.psi1_f, jtfs.psi2_f]):
                  for k in psi_fs[-1]:
                      if not isinstance(k, int):
                          continue
                      # check psis only
                      lp = np.sum([np.abs(p[k])**2 for p in psi_fs if k in p],
                                  axis=0)

                      check_above(lp, test_params_str, k=k)
                      check_below(lp, test_params_str, psi_fs, k=k)

              # frequential filterbank
              for s1_fr, psi_fs in enumerate([jtfs.psi1_f_fr_up,
                                              jtfs.psi1_f_fr_down]):
                  j0_max = max(j for p in psi_fs for j in p if isinstance(j, int))
                  # check per j0
                  for j0 in range(j0_max):
                      lp = 0
                      for p in psi_fs:
                          if j0 in p:
                              lp += np.abs(p[j0])**2
                      check_above(lp, test_params_str, j0=j0)
                      check_below(lp, test_params_str, psi_fs, j0=j0)
              # assert same peak values (symmetry)
              for i, (p_up, p_dn) in enumerate(zip(jtfs.psi1_f_fr_up,
                                                   jtfs.psi1_f_fr_down)):
                  up_mx, dn_mx = p_up[0].max(), p_dn[0].max()
                  assert up_mx == dn_mx, (i, up_mx, dn_mx,
                                          "\n%s" % test_params_str)


def test_compute_temporal_width():
    """Tests that `compute_temporal_width` works as intended."""
    # library defaults
    sigma0 = .1
    criterion_amplitude = 1e-3
    complete_decay_factor = 16  # follows from above

    J_pad = 9
    filter_len = 2**J_pad
    pts_per_scale = 6
    # don't allow underestimating by more than this
    th_undershoot = -1
    # consider `T` above this as close to global averaging
    T_global_avg_earliest = int(.6 * filter_len // 2)
    T_global_avg_latest = int(.8 * filter_len // 2)

    Ts = np.arange(2, 256)
    # test for different input sizes relative to filter sizes
    for N in (filter_len, filter_len // 2, filter_len // 4):
        T_ests = []
        for T in Ts:
            phi_f = gauss_1d(filter_len, sigma=sigma0/T)
            if T > N // 2:
                break
            T_est = compute_temporal_width(
                phi_f, N, sigma0=sigma0, criterion_amplitude=criterion_amplitude,
                pts_per_scale=pts_per_scale)
            T_ests.append(T_est)
        Ts = Ts[:len(T_ests)]
        T_ests = np.array(T_ests)

        for (T, T_est) in zip(Ts, T_ests):
            test_params_str = 'T={}, N={}, T_est={}'.format(T, N, T_est)

            # check global averaging cases
            if N == filter_len:
                if T_est == N:
                    assert T >= T_global_avg_earliest, "{} < {} | {}".format(
                        T, T_global_avg_earliest, test_params_str)
                elif T >= T_global_avg_latest:
                    assert T_est == N, "{} != {} | {}".format(
                        T_est, N, test_params_str)
            elif T == Ts[-1]:
                # last is max
                assert T_est == T_ests.max(), "{} != {} | {}".format(
                    T_est, T_ests.max(), test_params_str)

            # check other cases
            complete_decay = bool(T <= filter_len // complete_decay_factor)
            if complete_decay:
                # must match perfectly
                assert T_est == T, "{} != {} | {}".format(
                    T_est, T, test_params_str)
            else:
                assert T_est - T > th_undershoot, "{} - {} <= {} | {}".format(
                    T_est, T, th_undershoot, test_params_str)

    # Agreement of `fast=True` with `False` for complete decay
    # also try non-default `sigma0`
    N = 256
    for T in range(1, 16):
        p_f = gauss_1d(N, sigma=0.1 / T)
        w0 = compute_temporal_width(p_f, fast=True)
        w1 = compute_temporal_width(p_f, fast=False)
        assert w0 == w1 == T, (w0, w1, T)

        sigma0 = .15
        p_f = gauss_1d(N, sigma=sigma0 / T)
        w = compute_temporal_width(p_f, sigma0=sigma0, fast=False)
        assert w == T, (w, T)


def test_tensor_padded():
    """Test `tensor_padded` works as intended."""
    ls = [[[1, 2, 3, 4],
           [1, 2, 3],],
          [[1, 2, 3],
           [1, 2],
           [1],],
         ]
    target = np.array([[[1, 2, 3, 4],
                        [1, 2, 3, 0],
                        [0, 0, 0, 0]],
                       [[1, 2, 3, 0],
                        [1, 2, 0, 0],
                        [1, 0, 0, 0]]])
    out = tensor_padded(ls)
    assert np.all(target == out), out

    # with `pad_value`
    target[target == 0] = -2
    out = tensor_padded(ls, pad_value=-2)
    assert np.all(target == out), out


def test_pack_coeffs_jtfs():
    """Test coefficients are packed as expected."""
    def out_stored_into_pairs(out_stored, out_stored_keys):
        paired_flat = {}
        for i, k in enumerate(out_stored_keys):
            pair = k.split(':')[0]
            if pair not in paired_flat:
                paired_flat[pair] = []
            paired_flat[pair].append({'coef': out_stored[i]})
        return paired_flat

    def validate_n2s(o, info, spin):
        info = info + "\nspin={}".format(spin)
        # ensure 4-dim
        assert o.ndim == 4, "{}{}".format(o.shape, info)

        # pack here directly via arrays, see if they match
        n2s = o[:, :, :, 0]

        # fetch unpadded
        n2s_no_pad = n2s[np.where(n2s != -2)]
        # if phi_t is present, ensure it's the first (and only the first)
        if -1 in n2s:
            n2s0_no_pad = n2s[:1][np.where(n2s[:1] != -2)]
            assert np.all(n2s0_no_pad == -1), "{}{}".format(n2s, info)
            assert -1 not in n2s[1:], "{}{}".format(n2s, info)

        # should never require to pad along `n2` (i.e. fully empty coeff)
        for o_n2 in o:
            assert not np.all(o_n2 == -2), "{}{}".format(o_n2, info)

        # exclude phis
        n2s = n2s[n2s != -1]

        # ensure high-to-low n2 (low-to-high freq)
        # no pad & no lowpass
        n2s_np_nlp = n2s_no_pad[np.where(n2s_no_pad != -1)]
        assert np.all(n2s_np_nlp == sorted(n2s_np_nlp, reverse=True)
                      ), "{}{}".format(n2s, info)

    def validate_n1s(o, info, spin):
        info = info + "\nspin={}".format(spin)
        # ensure n1s ordered low to high (high-to-low freq) for every n2, n1_fr
        for n2_idx in range(len(o)):
            for n1_fr_idx in range(len(o[n2_idx])):
                n1s = o[n2_idx, n1_fr_idx, :, 2]
                if -2 in n1s:
                    # assert left-padded
                    n_pad = sum(n1s == -2)
                    assert np.all(n1s[:n_pad] == -2), (
                        "{}, {}{}").format(n_pad, n1s, info)
                # remove padded
                n1s = np.array([n1 for n1 in n1s if n1 != -2])
                assert np.all(n1s == sorted(n1s, reverse=True)), (
                    "{}, {}, {}{}").format(n2_idx, n1_fr_idx, n1s, info)

    def validate_spin(out_s, info, up=True):
        info = info + "\nspin={}".format("up" if up else "down")
        # check every n1_fr
        for n2_idx in range(len(out_s)):
            n1_frs = out_s[n2_idx, :, 0, 1]

            # if phi_f is present, ensure it's centered (and len(n1_frs) is odd)
            if -1 in n1_frs:
                if up:
                    assert n1_frs[-1] == -1, "%s\n%s" % (n1_frs, info)
                else:
                    assert n1_frs[0] == -1, "%s\n%s" % (n1_frs, info)
                # exclude phi_f
                n1_frs = np.array([n1_fr for n1_fr in n1_frs if n1_fr != -1])

            # ensure only psi_f pairs present
            assert -1 not in n1_frs, (n1_frs, info)

            # check padding
            if -2 in n1_frs:
                n_pad = sum(n1_frs == -2)
                if up:
                    # ensure right-padded
                    assert np.all(n1_frs[-n_pad:] == -2), "%s, %s\n%s" % (
                        n_pad, n1_frs, info)
                else:
                    # ensure left-padded
                    assert np.all(n1_frs[:n_pad]  == -2), "%s, %s\n%s" % (
                        n_pad, n1_frs, info)
                # exclude pad values
                n1_frs = np.array([n1_fr for n1_fr in n1_frs if n1_fr != -2])

            errmsg = "{}{}".format(n1_frs, info)
            if up:
                assert np.all(n1_frs == sorted(n1_frs)), errmsg
            else:
                assert np.all(n1_frs == sorted(n1_frs, reverse=True)), errmsg

    def validate_packing(out, separate_lowpass, structure, t, info):
        # unpack into `out_up, out_down, out_phi`
        out_phi_f, out_phi_t = None, None
        if structure in (1, 2):
            if separate_lowpass:
                out, out_phi_f, out_phi_t = out

            if structure == 1:
                out = out.transpose(1, 0, 2, 3)
                if separate_lowpass:
                    if out_phi_f is not None:
                        out_phi_f = out_phi_f.transpose(1, 0, 2, 3)
                    if out_phi_t is not None:
                        out_phi_t = out_phi_t.transpose(1, 0, 2, 3)

            s = out.shape
            out_up   = (out[:, :s[1]//2 + 1] if not separate_lowpass else
                        out[:, :s[1]//2])
            out_down = out[:, s[1]//2:]

        elif structure == 3:
            if not separate_lowpass:
                out_up, out_down, out_phi_f = out
            else:
                out_up, out_down, out_phi_f, out_phi_t = out

        elif structure == 4:
            if not separate_lowpass:
                out_up, out_down = out
            else:
                out_up, out_down, out_phi_t = out

        # ensure sliced properly
        assert out_up.shape == out_down.shape, (
            "{} != {}{}").format(out_up.shape, out_down.shape, info)

        # do validation ######################################################
        # n1s and n2s
        outs = (out_up, out_down, out_phi_f, out_phi_t)
        for spin, o in zip([1, -1, 0, 0], outs):
            if o is not None:
                assert o.shape[-1] == t, (o.shape, t)
                validate_n2s(o, info, spin)
                validate_n1s(o, info, spin)

        # n1_frs
        validate_spin(out_up,   info, up=True)
        validate_spin(out_down, info, up=False)

        # `phi_f`
        if out_phi_f is not None:
            out_phi_f_n1_fr = out_phi_f[:, :, :, 1]
            # exclude pad
            out_phi_f_n1_fr[out_phi_f_n1_fr == -2] = -1
            assert np.all(out_phi_f_n1_fr == -1), (out_phi_f_n1_fr, info)

        # `phi_t`
        if out_phi_t is not None:
            out_phi_t_n2 = out_phi_t[:, :, :, 0]
            # exclude pad
            out_phi_t_n2[out_phi_t_n2 == -2] = -1
            assert np.all(out_phi_t_n2 == -1), (out_phi_t_n2, info)

    # end of helper methods ##################################################
    # test
    tests_params = {
        1: dict(average=True, average_fr=True,  aligned=True,  out_3D=True),
        0: dict(average=True, average_fr=True,  aligned=False, out_3D=False),
        4: dict(average=True, average_fr=False, aligned=True,  out_3D=False),
    }

    for test_num, test_params in tests_params.items():
        _, out_stored, out_stored_keys, params, _, meta = load_data(test_num)
        t = out_stored[0].shape[-1]

        # ensure match
        for k in test_params:
            if k != 'average':
                assert test_params[k] == params[k]
        test_params['sampling_psi_fr'] = ('resample' if test_num != 0 else
                                          'exclude')

        # flatten rather than re-pack into original shape since it's flattened
        # in `pack_coeffs_jtfs` anyway
        paired_flat0 = out_stored_into_pairs(out_stored, out_stored_keys)

        for separate_lowpass in (False, True):
          for structure in (1, 2, 3, 4):
            for out_exclude in (None, 1):
                if out_exclude is not None:
                    if not separate_lowpass:
                        # invalid
                        continue
                    else:
                        pairs_lp = ('psi_t * phi_f', 'phi_t * psi_f',
                                    'phi_t * phi_f')
                        out_exclude = (pairs_lp[-3:] if structure != 4 else
                                       pairs_lp[-2:])

                if not (structure == 1 and
                        not separate_lowpass and
                        test_params['average'] and
                        test_params['average_fr'] and
                        not test_params['out_3D'] and
                        test_params['sampling_psi_fr'] == 'exclude'):
                    continue
                kw = {k: v for k, v in test_params.items()
                      if k in ('sampling_psi_fr',)}
                info = "\nstructure={}\nseparate_lowpass={}\n{}".format(
                    structure, separate_lowpass,
                    "\n".join(f'{k}={v}' for k, v in test_params.items()))

                # exclude pairs if needed
                if out_exclude is None:
                    paired_flat = paired_flat0
                else:
                    paired_flat = {}
                    for pair in paired_flat0:
                        if pair not in out_exclude:
                            paired_flat[pair] = paired_flat0[pair]

                # pack & validate
                out = pack_coeffs_jtfs(paired_flat, meta, structure=structure,
                                       separate_lowpass=separate_lowpass,
                                       out_3D=test_params['out_3D'],
                                       **kw, debug=True)
                validate_packing(out, separate_lowpass, structure, t, info)


def test_energy_conservation():
    """E_out ~= E_in.

    Refer to:
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-864097
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855701
     - https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855702
    """
    np.random.seed(0)
    # 8.5 on dyadic scale to test time unpad correction
    N = 360
    x = np.random.randn(N)

    # configure ~tight frame; configured also for speed, not optimized for I/O
    # https://github.com/kymatio/kymatio/discussions/732#discussioncomment-855703
    # https://github.com/kymatio/kymatio/discussions/732#discussioncomment-968384
    r_psi = (.9, .9, .9)
    J = int(np.ceil(np.log2(N)) - 2)
    J_fr = 3
    F = 2**J_fr
    T = 2**J
    Q = (8, 3)
    Q_fr = 4
    params = dict(
        shape=N, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr, F=F, T=T,
        r_psi=r_psi, average_fr=False, sampling_filters_fr='resample',
        max_pad_factor=None, max_pad_factor_fr=None,
        pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
        out_type='dict:list', frontend=default_backend
    )
    jtfs_a = TimeFrequencyScattering1D(**params, average=True)
    jtfs_u = TimeFrequencyScattering1D(**params, average=False)
    jmeta_a = jtfs_a.meta()
    jmeta_u = jtfs_u.meta()

    # scatter
    Scx_a = jtfs_a(x)
    Scx_u = jtfs_u(x)

    # compute energies
    pairs = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_down')
    kw = dict(kind='l2', plots=False)
    _, pair_energies_a = energy_profile_jtfs(Scx_a, jmeta_a, **kw, pairs=pairs)
    _, pair_energies_u = energy_profile_jtfs(Scx_u, jmeta_u, **kw, pairs=pairs)

    pe_a, pe_u = [{pair: np.sum(pe[pair]) for pair in pe}
                  for pe in (pair_energies_a, pair_energies_u)]

    # compute energy relations ###############################################
    E = {}
    E['in'] = energy(x)
    E['out'] = (np.sum([v for pair, v in pe_u.items()
                        if pair not in ('S0', 'S1')]) +
                pe_a['S0'])

    E['S0'] = pe_a['S0']
    E['S1'] = pe_a['S1']
    E['U1'] = pe_u['S1']
    # U2 + S2 = U1 - S1 --> U2 = (U1 - S1) - S2
    E['U2'] = (E['U1'] - E['S1']) - pe_u['psi_t * phi_f']
    E['S1_joint'] = (pe_u['phi_t * phi_f'] +
                     pe_u['phi_t * psi_f'])
    E['U2_joint'] = (pe_u['psi_t * psi_f_up'] +
                     pe_u['psi_t * psi_f_down'])

    r = {}
    r['out / in'] = E['out'] / E['in']
    r['(S0 + U1) / in'] = (E['S0'] + E['U1']) / E['in']
    r['S1_joint / S1'] = E['S1_joint'] / E['S1']
    r['U2_joint / U2'] = E['U2_joint'] / E['U2']

    if metric_verbose:
        print("\nEnergy conservation (w/ tight frame):")
        for k, v in r.items():
            print("{:.4f} -- {}".format(v, k))

    # run assertions #########################################################
    assert .9  < r['out / in']       < 1., r['out / in']
    assert .95 < r['(S0 + U1) / in'] < 1., r['(S0 + U1) / in']
    assert .93 < r['S1_joint / S1']  < 1., r['S1_joint / S1']
    assert .83 < r['U2_joint / U2']  < 1., r['U2_joint / U2']


def test_est_energy_conservation():
    """Tests that `toolkit.est_energy_conservation` doesn't error, and that
    values are sensible.
    """
    N = 256
    x = np.random.randn(N)

    kw = dict(verbose=1, analytic=1)
    print()
    ESr0 = est_energy_conservation(x, jtfs=0, **kw)
    print()
    ESr1 = est_energy_conservation(x, jtfs=1, **kw)

    for ESr in (ESr0, ESr1):
        for k, v in ESr.items():
            tol = .03  # random with torch cuda default
            assert 0 < v < 1 + tol, (k, v)


def test_implementation():
    """Test that every `implementation` kwarg works."""
    N = 512
    x = echirp(N)

    for implementation in range(1, 6):
        jtfs = TimeFrequencyScattering1D(shape=N, J=4, Q=2,
                                         implementation=implementation,
                                         frontend=default_backend)
        _ = jtfs(x)


def test_pad_mode_fr():
    """Test that functional `pad_mode_fr` works."""
    from kymatio.scattering1d.core.timefrequency_scattering1d import _right_pad
    N = 512
    x = echirp(N)

    kw = dict(shape=N, J=4, Q=2, frontend=default_backend, out_type='array',
              max_pad_factor_fr=1)
    jtfs0 = TimeFrequencyScattering1D(**kw, pad_mode_fr='zero')
    jtfs1 = TimeFrequencyScattering1D(**kw, pad_mode_fr=_right_pad)

    out0 = jtfs0(x)
    out1 = jtfs1(x)
    assert np.allclose(out0, out1)


def test_normalize():
    """Ensure error thrown upon invalid input, but otherwise method
    doesn't error.
    """
    for rscaling in ('l1', 'l2'):
      for mean_axis in (0, (1, 2), -1):
        for std_axis in (0, (1, 2), -1):
          for C in (None, 2):
            for mu in (None, 2):
              for dim0 in (1, 64):
                for dim1 in (1, 65):
                  for dim2 in (1, 66):
                      if dim0 == dim1 == dim2 == 1:
                          # invalid combo
                          continue
                      x = np.abs(np.random.randn(dim0, dim1, dim2))
                      test_params = dict(
                          rscaling=rscaling, mean_axis=mean_axis,
                          std_axis=std_axis, C=C, mu=mu, dim0=dim0, dim1=dim1,
                          dim2=dim2)
                      test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                                  test_params.items())
                      try:
                          kw = {k: v for k, v in test_params.items()
                                if k not in ('dim0', 'dim1', 'dim2')}
                          _ = normalize(x, **kw)
                      except ValueError as e:
                          if "input dims cannot be" in str(e):
                              continue
                          else:
                              print(test_params_str)
                              raise e
                      except Exception as e:
                          print(test_params_str)
                          raise e


def test_no_second_order_filters():
    """Reproduce edge case: configuration yields no second-order wavelets
    so can't do JTFS.
    """
    with pytest.raises(ValueError) as record:
        _ = TimeFrequencyScattering1D(shape=8192, J=1, Q=2, r_psi=.9,
                                      frontend=default_backend)
    assert "no second-order filters" in record.value.args[0]


def test_backends():
    for backend_name in ('torch', 'tensorflow'):
        if cant_import(backend_name):
            continue
        elif backend_name == 'torch':
            import torch
        elif backend_name == 'tensorflow':
            import tensorflow as tf

        N = 2048
        x = echirp(N)
        x = np.vstack([x, x, x])
        x = (tf.constant(x) if backend_name == 'tensorflow' else
             torch.from_numpy(x))

        jtfs = TimeFrequencyScattering1D(shape=N, J=8, Q=8, J_fr=3, Q_fr=1,
                                         average_fr=True, out_type='dict:array',
                                         out_3D=True, frontend=backend_name)
        Scx = jtfs(x)
        jmeta = jtfs.meta()

        # test batched packing for convenience ###############################
        for structure in (1, 2, 3, 4):
            for separate_lowpass in (False, True):
                kw = dict(meta=jmeta, structure=structure,
                          separate_lowpass=separate_lowpass,
                          sampling_psi_fr=jtfs.sampling_psi_fr)
                # keep original copy
                Scxnc  = deepcopy(jtfs_to_numpy(Scx))
                outs   = pack_coeffs_jtfs(Scx, **kw)
                outs0  = pack_coeffs_jtfs(Scx, **kw, sample_idx=0)
                outsn  = pack_coeffs_jtfs(jtfs_to_numpy(Scx), **kw)
                outs0n = pack_coeffs_jtfs(jtfs_to_numpy(Scx), **kw, sample_idx=0)
                outs   = outs   if isinstance(outs,  tuple) else [outs]
                outs0  = outs0  if isinstance(outs0, tuple) else [outs0]
                outsn  = outsn  if isinstance(outs,  tuple) else [outsn]
                outs0n = outs0n if isinstance(outs0, tuple) else [outs0n]

                # ensure methods haven't altered original array ##############
                # (with e.g. inplace ops) ####################################
                for pair in Scx:
                    coef = Scx[pair].cpu().numpy()
                    assert np.allclose(coef, Scxnc[pair]), pair

                # shape and value checks #####################################
                for o, o0, on, o0n in zip(outs, outs0, outsn, outs0n):
                    assert o.ndim == 5, o.shape
                    assert len(o) == len(x), (len(o), len(x))
                    assert o.shape[-1] == Scx['S0'].shape[-1], (
                        o.shape, Scx['S0'].shape)
                    assert o.shape[1:] == o0.shape, (o.shape, o0.shape)
                    assert np.allclose(o.numpy(), on)
                    assert np.allclose(o0.numpy(), o0n)

                # E_in == E_out ##############################################
                _test_packing_energy_io(Scx, outs, structure, separate_lowpass)

        ######################################################################

        Scx = jtfs_to_numpy(Scx)
        E_up   = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_up')
        E_down = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_down')
        th = 32
        assert E_down / E_up > th, "{:.2f} < {}".format(E_down / E_up, th)


def _test_packing_energy_io(Scx, outs, structure, separate_lowpass):
    from kymatio.toolkit import ExtendedUnifiedBackend

    def phi_t_energies(out_phi_t, structure):
        if structure == 1:
            B = ExtendedUnifiedBackend(out_phi_t)
            out_phi_t = B.transpose(out_phi_t, (0, 2, 1, 3, 4))
        s = tuple(out_phi_t.shape)
        # out_phi_t includes `phi_t * phi_f`
        E_out_phi = energy(out_phi_t[:, :, s[2]//2])
        E_out_phi_t = energy(out_phi_t) - E_out_phi
        return E_out_phi, E_out_phi_t

    def check_phi_t_energies(out_phi_t, structure):
        E_out_phi, E_out_phi_t = phi_t_energies(out_phi_t, structure)
        assert np.allclose(E_in_phi_t, E_out_phi_t), (E_in_phi_t, E_out_phi_t)
        assert np.allclose(E_in_phi, E_out_phi), (E_in_phi, E_out_phi)
        return E_out_phi, E_out_phi_t

    E_in_up = energy(Scx['psi_t * psi_f_up'])
    E_in_dn = energy(Scx['psi_t * psi_f_down'])
    E_in_spinned = E_in_up + E_in_dn
    E_in_phi_t = energy(Scx['phi_t * psi_f'])
    E_in_phi_f = energy(Scx['psi_t * phi_f'])
    E_in_phi   = energy(Scx['phi_t * phi_f'])
    E_in = E_in_spinned + E_in_phi_t + E_in_phi_f + E_in_phi
    if structure in (1, 2):
        if separate_lowpass:
            out_spinned, out_phi_f, out_phi_t = outs
            E_out_spinned = energy(out_spinned)
            assert np.allclose(E_in_spinned, E_out_spinned
                               ), (E_in_spinned, E_out_spinned)

            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure)

            E_out_phi_f = energy(out_phi_f)
            assert np.allclose(E_in_phi_f, E_out_phi_f), (E_in_phi_f, E_out_phi_f)

            E_out = E_out_spinned + E_out_phi_f + E_out_phi + E_out_phi_t
        else:
            E_out = energy(outs[0])

    elif structure == 3:
        if separate_lowpass:
            """E(outs) == E_in_spinned + E_in_phi_t + E_in_phi_f + 2*E_in_phi"""
            out_up, out_down, out_phi_f, out_phi_t = outs

            # method properly splits `phi_t * phi_f` and `phi_t * psi_f` energies
            # so no duplication in `E_out`
            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure)
        else:
            """E(outs) == E_in_spinned + E_in_phi_t + E_in_phi_f + 2*E_in_phi"""
            out_up, out_down, out_phi_f = outs
            # `phi_t * psi_f` packed along `psi_t * psi_f`
            E_in_spinned += E_in_phi_t

        E_out_spinned = energy(out_up) + energy(out_down)
        assert np.allclose(E_in_spinned, E_out_spinned
                           ), (E_in_spinned, E_out_spinned)

        E_out_phi_f = energy(out_phi_f)
        # `phi_t * phi_f` is packed with `psi_t * phi_f`
        E_in_phi_f += E_in_phi
        assert np.allclose(E_in_phi_f, E_out_phi_f), (E_in_phi_f, E_out_phi_f)

        E_out = E_out_spinned + E_out_phi_f
        if separate_lowpass:
            E_out += E_out_phi_t

    elif structure == 4:
        if separate_lowpass:
            out_up, out_down, out_phi_t = outs
            # `psi_t * phi_f` packed for each spin
            E_in_spinned += 2*E_in_phi_f
            E_out_spinned = energy(out_up) + energy(out_down)

            E_out_phi, E_out_phi_t = check_phi_t_energies(out_phi_t, structure)
        else:
            out_up, out_down = outs
            # phi_t and phi_f pairs included
            # only `phi_t * phi_f` and `psi_t * phi_f` energy duped
            E_in_spinned += E_in_phi_t + 2*(E_in_phi + E_in_phi_f)
            E_out_spinned = energy(out_up) + energy(out_down)

        assert np.allclose(E_in_spinned, E_out_spinned
                           ), (E_in_spinned, E_out_spinned)

        E_out = E_out_spinned
        E_in = E_in_spinned
        if separate_lowpass:
            E_out += (E_out_phi + E_out_phi_t)
            E_in  += (E_in_phi + E_in_phi_t)

    assert np.allclose(E_in, E_out), (E_in, E_out, structure, separate_lowpass)


def test_differentiability_torch():
    """Tests whether JTFS is differentiable in PyTorch backend.
    Does NOT test whether the gradients are correct.
    """
    if cant_import('torch'):
        return
    import torch
    if torch.cuda.is_available():
        devices = ['cuda', 'cpu']
    else:
        devices = ['cpu']

    J = 6
    Q = 8
    N = 2**12
    for device in devices:
        jtfs = TimeFrequencyScattering1D(J, N, Q, frontend='torch',
                                         out_type='array', max_pad_factor=1
                                         ).to(device)
        x = torch.randn(2, N, requires_grad=True, device=device)

        s = jtfs.forward(x)
        loss = torch.sum(torch.abs(s))
        loss.backward()
        assert torch.max(torch.abs(x.grad)) > 0.


def test_reconstruction_torch():
    """Test that input reconstruction via backprop has decreasing loss."""
    if cant_import('torch'):
        return
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    J = 6
    Q = 8
    N = 1024
    n_iters = 30
    jtfs = TimeFrequencyScattering1D(J, N, Q, J_fr=4,
                                     frontend='torch', out_type='array',
                                     sampling_filters_fr=('exclude', 'resample'),
                                     max_pad_factor=1, max_pad_factor_fr=2
                                     ).to(device)
    jtfs.meta()

    y = torch.from_numpy(echirp(N, fmin=1).astype('float32')).to(device)
    Sy = jtfs(y)
    div = Sy.max()
    Sy /= div

    torch.manual_seed(0)
    x = torch.randn(N, device=device)
    x /= torch.max(torch.abs(x))
    x.requires_grad = True
    optimizer = torch.optim.SGD([x], lr=140000, momentum=.9, nesterov=True)
    loss_fn = torch.nn.MSELoss()

    losses, losses_recon = [], []
    for i in range(n_iters):
        optimizer.zero_grad()
        Sx = jtfs(x)
        Sx /= div
        loss = loss_fn(Sx, Sy)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().numpy()))
        xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        losses_recon.append(float(rel_l2(yn, xn)))

    th, th_recon, th_end_ratio = 1e-5, 1.05, 50
    end_ratio = losses[0] / losses[-1]
    assert end_ratio > th_end_ratio, end_ratio
    assert min(losses) < th, "{:.2e} > {}".format(min(losses), th)
    assert min(losses_recon) < th_recon, "{:.2e} > {}".format(min(losses_recon),
                                                              th_recon)
    if metric_verbose:
        print(("\nReconstruction (torch):\n(end_start_ratio, min_loss, "
               "min_loss_recon) = ({:.1f}, {:.2e}, {:.2f})").format(
                   end_ratio, min(losses), min(losses_recon)))


def test_meta():
    """Tests that meta values and structures match those of output for all
    combinations of
        - out_3D (True only with average_fr=True and sampling_psi_fr != 'exclude')
        - average_fr
        - average
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
    and, partially
        - average_global
        - average_global_phi
        - average_global_fr
        - average_global_fr_phi
    a total of 60 tests. All possible ways of packing the same coefficients
    (via `out_type`) aren't tested.

    Not tested:
        - oversampling_fr
        - max_padding_fr

    For compute convenience, also tests that `validate_filterbank_tm` and
    `validate_filterbank_fr` run without error on each configuration.
    """
    def assert_equal_lengths(Scx, jmeta, field, pair, out_3D, test_params_str,
                             jtfs):
        """Assert that number of coefficients and frequency rows for each match"""
        if out_3D:
            out_n_coeffs  = len(Scx[pair])
            out_n_freqs   = sum(len(c['coef'][0]) for c in Scx[pair])
            meta_n_coeffs = len(jmeta[field][pair])
            meta_n_freqs  = np.prod(jmeta[field][pair].shape[:2])

            assert out_n_coeffs == meta_n_coeffs, (
                "len(out[{0}]), len(jmeta[{1}][{0}]) = {2}, {3}\n{4}"
                ).format(pair, field, out_n_coeffs, meta_n_coeffs,
                         test_params_str)
        else:
            out_n_freqs  = sum(c['coef'].shape[1] for c in Scx[pair])
            meta_n_freqs = len(jmeta[field][pair])

        assert out_n_freqs == meta_n_freqs, (
            "out vs meta n_freqs mismatch for {}, {}: {} != {}\n{}".format(
                pair, field, out_n_freqs, meta_n_freqs, test_params_str))

    def assert_equal_values(Scx, jmeta, field, pair, i, meta_idx, out_3D,
                            test_params_str, test_params, jtfs):
        """Assert that non-NaN values are equal."""
        a, b = Scx[pair][i][field], jmeta[field][pair][meta_idx[0]]
        errmsg = ("(out[{0}][{1}][{2}], jmeta[{2}][{0}][{3}]) = ({4}, {5})\n{6}"
                  ).format(pair, i, field, meta_idx[0], a, b, test_params_str)

        meta_len = b.shape[-1]
        zeroth_order_unaveraged = (pair == 'S0' and not test_params['average'])
        if field not in ('s', 'stride'):
            assert meta_len == 3, ("all meta fields (except spin, stride) must "
                                   "pad to length 3: %s" % errmsg)
            if not zeroth_order_unaveraged:
                assert len(a) > 0, ("all computed metas (except spin) must "
                                    "append something: %s" % errmsg)

        if field == 'stride':
            assert meta_len == 2, ("'stride' meta length must be 2 "
                                   "(got meta: %s)" % b)
            if pair in ('S0', 'S1'):
                if pair == 'S1' or test_params['average']:
                    assert len(a) == 1, errmsg
                if pair == 'S0' and not test_params['average']:
                    assert a == (), errmsg
                    assert np.all(np.isnan(b)), errmsg
                else:
                    assert a == b[..., 1], errmsg
                    assert np.isnan(b[..., 0]), errmsg
            else:
                assert len(a) == 2, errmsg
                assert np.all(a == b), errmsg
                assert not np.any(np.isnan(b)), errmsg

        elif (field == 's' and pair in ('S0', 'S1')) or zeroth_order_unaveraged:
            assert len(a) == 0 and np.all(np.isnan(b)), errmsg

        elif len(a) == meta_len:
            assert np.all(a == b), errmsg

        elif len(a) < meta_len:
            # S0 & S1 have one meta entry per coeff so we pad to joint's len
            if np.all(np.isnan(b[:2])):
                assert pair in ('S0', 'S1'), errmsg
                assert a[0] == b[..., 2], errmsg
            # joint meta is len 3 but at compute 2 is appended
            elif len(a) == 2 and meta_len == 3:
                assert pair not in ('S0', 'S1'), errmsg
                assert np.all(a[:2] == b[..., :2]), errmsg

        else:
            # must meet one of above behaviors
            raise AssertionError(errmsg)

        # increment meta_idx for next check
        if pair in ('S0', 'S1') or out_3D:
            meta_idx[0] += 1
        else:
            # increment by number of frequential rows (i.e. `n1`) since
            # these n1-meta aren't appended in computation
            n_freqs = Scx[pair][i]['coef'].shape[1]
            meta_idx[0] += n_freqs

    def assert_aligned_stride(Scx, test_params_str):
        """Assert all frequential strides are equal in `aligned`."""
        ref_stride = Scx['psi_t * psi_f_up'][0]['stride'][0]
        for pair in Scx:
            if pair in ('S0', 'S1'):
                continue
            for i, c in enumerate(Scx[pair]):
                s = c['stride'][0]
                assert s == ref_stride, (
                    "Scx[{}][{}]['stride'] = {} != ref_stride == {}\n{}".format(
                        pair, i, s, ref_stride, test_params_str))

    def run_test(params, test_params):
        jtfs = TimeFrequencyScattering1D(**params, **test_params,
                                         frontend=default_backend)
        test_params_str = '\n'.join(f'{k}={v}' for k, v in test_params.items())
        _ = validate_filterbank_tm(jtfs, verbose=0)
        _ = validate_filterbank_fr(jtfs, verbose=0)

        sampling_psi_fr = test_params['sampling_filters_fr'][0]
        if sampling_psi_fr in ('recalibrate', 'exclude'):
            # assert not all J_pad_fr are same so test covers this case
            # psi is dominant here as `2**J_fr > F`
            assert_pad_difference(jtfs, test_params_str)

        try:
            Scx = jtfs(x)
            jmeta = jtfs.meta()
        except Exception as e:
            print("Failed at:\n%s" % test_params_str)
            raise e

        # ensure no output shape was completely reduced
        for pair in Scx:
            for i, c in enumerate(Scx[pair]):
                assert not np.any(c['coef'].shape == 0), (pair, i)

        # meta test
        out_3D = test_params['out_3D']
        for field in ('j', 'n', 's', 'stride'):
          for pair in jmeta[field]:
            meta_idx = [0]
            assert_equal_lengths(Scx, jmeta, field, pair, out_3D,
                                 test_params_str, jtfs)
            for i in range(len(Scx[pair])):
                assert_equal_values(Scx, jmeta, field, pair, i, meta_idx,
                                    out_3D, test_params_str, test_params, jtfs)

        # check stride if `aligned`
        if aligned:
            assert_aligned_stride(Scx, test_params_str)

        # save compute and test this method for thoroughness
        if average:
            for structure in (1, 2, 3, 4):
                for separate_lowpass in (False, True):
                    try:
                        _ = pack_coeffs_jtfs(Scx, jmeta, structure=structure,
                                             separate_lowpass=separate_lowpass,
                                             sampling_psi_fr=jtfs.sampling_psi_fr,
                                             out_3D=jtfs.out_3D)
                    except Exception as e:
                        print(test_params_str)
                        raise e

    if default_backend != 'numpy':
        # meta doesn't change
        warnings.warn("`test_meta()` skipped per non-'numpy' `default_backend`")
        return

    N = 512
    x = np.random.randn(N)

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = (16, 1)
    J_fr = 5
    Q_fr = 2
    F = 4
    out_type = 'dict:list'
    params = dict(shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type)

    for out_3D in (False, True):
      for average_fr in (True, False):
        for average in (True, False):
          if out_3D and not (average_fr and average):
              continue  # invalid option
          for aligned in (True, False):
            for sampling_psi_fr in ('resample', 'exclude', 'recalibrate'):
              for sampling_phi_fr in ('resample', 'recalibrate'):
                  test_params = dict(
                      out_3D=out_3D, average_fr=average_fr, average=average,
                      aligned=aligned,
                      sampling_filters_fr=(sampling_psi_fr, sampling_phi_fr))
                  run_test(params, test_params)

    # reproduce this case separately; above doesn't test where 'exclude' fails
    N = 1024
    x = np.random.randn(N)
    J = int(np.log2(N) - 1)
    J_fr = 3
    params = dict(shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type)

    sampling_psi_fr = 'exclude'
    out_3D = False
    for average_fr in (True, False):
      for average in (True, False):
        for aligned in (True, False):
          for sampling_phi_fr in ('resample', 'recalibrate'):
              test_params = dict(
                  out_3D=out_3D, average_fr=average_fr, average=average,
                  aligned=aligned,
                  sampling_filters_fr=(sampling_psi_fr, sampling_phi_fr))
              run_test(params, test_params)

    # minimal global averaging testing
    N = 512
    x = np.random.randn(N)
    params = dict(shape=N, J=9, Q=9, J_fr=5, Q_fr=1, F=2**5, out_type=out_type)
    for average_fr in (True, False):
        for average in (True, False):
            test_params = dict(average_fr=average_fr, average=average,
                               sampling_filters_fr='resample',
                               out_3D=False)
            run_test(params, test_params)


def test_output():
    """Applies JTFS on a stored signal to make sure its output agrees with
    a previously calculated version. Tests for:

          (aligned, average_fr, out_3D,   F,        sampling_filters_fr)
        0. False    True        False     32        ('exclude', 'recalibrate')
        1. True     True        True      4         ('resample', 'resample')
        2. False    True        True      16        ('resample', 'resample')
        3. True     True        False     'global'  ('resample', 'resample')
        4. True     False       False     8         ('resample', 'resample')
        5. False    True        True      16        ('recalibrate', 'recalibrate')

        6. special: params such that `scf.J_pad_fo > scf.J_pad_max`
            - i.e. all first-order coeffs pad to greater than longest set of
            second-order, as in `U1 * phi_t * phi_f` and
            `(U1 * phi_t * psi_f) * phi_t * phi_f`.

    For complete info see `data['code']` (`_load_data()`).
    """
    num_tests = sum((p.name.startswith('test_jtfs_') and p.suffix == '.npz')
                    for p in Path(test_data_dir).iterdir())

    for test_num in range(num_tests):
        if 0:#test_num != 0:
            continue
        (x, out_stored, out_stored_keys, params, params_str, _
         ) = load_data(test_num)

        jtfs = TimeFrequencyScattering1D(**params, frontend=default_backend)
        jmeta = jtfs.meta()
        out = jtfs(x)
        out = jtfs_to_numpy(out)

        # assert equal total number of coefficients
        if params['out_type'] == 'dict:list':
            n_coef_out = sum(len(o) for o in out.values())
            n_coef_out_stored = len(out_stored)
        elif params['out_type'] == 'dict:array':
            n_coef_out = sum(o.shape[1] for o in out.values())
            n_coef_out_stored = sum(len(o) for o in out_stored)
        assert n_coef_out == n_coef_out_stored, (
            "out vs stored number of coeffs mismatch ({} != {})\n{}"
            ).format(n_coef_out, n_coef_out_stored, params_str)

        i_s = 0
        mean_aes, max_aes = [0], [0]
        already_printed_test_info, max_mean_info, max_max_info = False, None, None
        for pair in out:
            for i, o in enumerate(out[pair]):
                n = jmeta['n'][pair][i]
                while n.squeeze().ndim > 1:
                    n = n[0]
                # assert equal shapes
                o = o if params['out_type'] == 'dict:array' else o['coef']
                o_stored, o_stored_key = out_stored[i_s], out_stored_keys[i_s]
                errmsg = ("out[{}][{}].shape != out_stored[{}].shape | n={}\n"
                          "({} != {})\n").format(pair, i, o_stored_key, n,
                                                 o.shape, o_stored.shape)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and o.shape != o_stored.shape:
                    # print(errmsg)
                    already_printed_test_info = True
                    i_s += 1
                    continue
                else:
                    assert o.shape == o_stored.shape, errmsg

                # store info for printing
                adiff = rel_ae(o_stored, o, ref_both=True)
                mean_ae, max_ae = adiff.mean(), adiff.max()
                if mean_ae > max(mean_aes):
                    max_mean_info = "out[%s][%s] | n=%s" % (pair, i, n)
                if max_ae > max(max_aes):
                    max_max_info  = "out[%s][%s] | n=%s" % (pair, i, n)
                mean_aes.append(mean_ae)
                max_aes.append(max_ae)

                # assert equal values
                errmsg = ("out[{}][{}] != out_stored[{}] | n={}\n"
                          "(MeanRAE={:.2e}, MaxRAE={:.2e})\n"
                          ).format(pair, i, o_stored_key, n,
                                   mean_aes[-1], max_aes[-1],)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and not np.allclose(o, o_stored):
                    # print(errmsg)
                    already_printed_test_info = True
                else:
                    assert np.allclose(o, o_stored), errmsg
                i_s += 1

        if output_test_print_mode:
            if max_mean_info is not None:
                print("// max_meanRAE = {:.2e} | {}\n".format(max(mean_aes),
                                                              max_mean_info))
            if max_max_info is not None:
                print("// max_maxRAE  = {:.2e} | {}\n".format(max(max_aes),
                                                              max_max_info))

### helper methods ###########################################################
def load_data(test_num):
    """Also see data['code']."""
    def is_meta(k):
        return k.startswith('meta:')
    def is_coef(k):
        return (':' in k and k.split(':')[-1].isdigit()) and not is_meta(k)
    def not_param(k):
        return k in ('code', 'x') or is_coef(k) or is_meta(k)

    data = np.load(Path(test_data_dir, f'test_jtfs_{test_num}.npz'))
    x = data['x']
    out_stored = [data[k] for k in data.files if is_coef(k)]
    out_stored_keys = [k for k in data.files if is_coef(k)]

    params = {}
    for k in data.files:
        if not_param(k):
            continue

        if k in ('average', 'average_fr', 'aligned'):
            params[k] = bool(data[k])
        elif k == 'sampling_filters_fr':
            params[k] = (bool(data[k]) if len(data[k]) == 1 else
                         tuple(data[k]))
        elif k == 'F':
            params[k] = (str(data[k]) if str(data[k]) == 'global' else
                         int(data[k]))
        elif k in ('out_type', 'pad_mode', 'pad_mode_fr'):
            params[k] = str(data[k])
        else:
            params[k] = int(data[k])

    meta = packed_meta_into_arr(data)

    params_str = "Test #%s:\n" % test_num
    for k, v in params.items():
        params_str += "{}={}\n".format(k, str(v))
    return x, out_stored, out_stored_keys, params, params_str, meta


def packed_meta_into_arr(data):
    meta_arr = {}
    for k in data.files:
        if not k.startswith('meta:'):
            continue
        _, field, pair, i = k.split(':')
        if field not in meta_arr:
            meta_arr[field] = {}
        if pair not in meta_arr[field]:
            meta_arr[field][pair] = []
        meta_arr[field][pair].append(data[k])

    for field in meta_arr:
        for pair in meta_arr[field]:
            meta_arr[field][pair] = np.array(meta_arr[field][pair])
    return meta_arr


def concat_joint(Scx):
    Scx = drop_batch_dim_jtfs(Scx)
    k = list(Scx)[0]
    out_type = ('list' if (isinstance(Scx[k], list) and 'coef' in Scx[k][0]) else
                'array')
    if out_type == 'array':
        return np.vstack([c for pair, c in Scx.items()
                          if pair not in ('S0', 'S1')])
    return np.vstack([c['coef'] for pair, coeffs in Scx.items()
                      for c in coeffs if pair not in ('S0', 'S1')])


def assert_pad_difference(jtfs, test_params_str):
    assert not all(
        J_pad_fr == jtfs.J_pad_frs_max
        for J_pad_fr in jtfs.J_pad_frs if J_pad_fr != -1
        ), "\n{}\nJ_pad_fr={}\nN_frs={}".format(
            test_params_str, jtfs.J_pad_frs, jtfs.N_frs)


if __name__ == '__main__':
    if run_without_pytest:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_up_vs_down()
        test_sampling_psi_fr_exclude()
        test_no_second_order_filters()
        test_max_pad_factor_fr()
        test_out_exclude()
        test_global_averaging()
        test_lp_sum()
        test_compute_temporal_width()
        test_tensor_padded()
        test_pack_coeffs_jtfs()
        test_energy_conservation()
        test_est_energy_conservation()
        test_implementation()
        test_pad_mode_fr()
        test_normalize()
        test_backends()
        test_differentiability_torch()
        test_reconstruction_torch()
        test_meta()
        test_output()
    else:
        pytest.main([__file__, "-s"])
