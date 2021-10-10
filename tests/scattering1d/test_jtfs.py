# -*- coding: utf-8 -*-
"""Joint Time-Frequency Scattering related tests."""
import pytest
import numpy as np
from pathlib import Path
from kymatio import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import (drop_batch_dim_jtfs, jtfs_to_numpy, coeff_energy,
                             fdts, echirp, coeff_energy_ratios, rel_l2)

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
    assert l2_jtfs / l2_ts > 15, ("\nJTFS/TS: %s \nTS: %s\nJTFS: %s"
                                  ) % (l2_jtfs / l2_ts, l2_ts, l2_jtfs)
    assert l2_ts < .008, "TS: %s" % l2_ts

    if metric_verbose:
        print(("\nFDTS sensitivity:\n"
               "JTFS/TS = {:.1f}\n"
               "TS      = {:.4f}\n").format(l2_jtfs / l2_ts, l2_ts))


def test_freq_tp_invar():
    """Test frequency transposition invariance -- omitted."""
    pass


def test_up_vs_down():
    """Test that echirp yields significant disparity in up vs down coeffs."""
    N = 2048
    x = echirp(N, fmin=64)

    if metric_verbose:
        print("\nFDTS directional sensitivity; E_down / E_up:")

    # values very different on main branch because `conj_reflections`
    # not implemented here
    m_th = (14, 420)
    l2_th = (14, 470)
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
        E_up, E_down = [list(E.values())[0] for E in (E_up, E_down)]
        r_l2 = E_down / E_up

        if metric_verbose:
            print(("Global:     {0:<5.1f} -- '{1}' pad\n"
                   "Slice mean: {2:<5.1f} -- '{1}' pad").format(
                       r_l2, pad_mode, r_m))
        assert r_l2 > l2_th[i], "{} < {} | '{}'".format(r_l2, l2_th[i], pad_mode)
        assert r_m  > m_th[i],  "{} < {} | '{}'".format(r_m,  m_th[i],  pad_mode)


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


def test_no_second_order_filters():
    """Reproduce edge case: configuration yields no second-order wavelets
    so can't do JTFS.
    """
    with pytest.raises(ValueError) as record:
        _ = TimeFrequencyScattering1D(shape=8192, J=1, Q=2, r_psi=.9,
                                      frontend=default_backend)
    assert "no second-order filters" in record.value.args[0]


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
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_up_vs_down()
        test_no_second_order_filters()
        test_max_pad_factor_fr()
        test_out_exclude()
        test_global_averaging()
        test_implementation()
        test_pad_mode_fr()
    else:
        pytest.main([__file__, "-s"])
