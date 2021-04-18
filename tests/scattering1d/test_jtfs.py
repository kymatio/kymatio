import pytest
import numpy as np
import scipy.signal
from kymatio.numpy import Scattering1D, TimeFrequencyScattering

# TODO no kymatio.numpy
# TODO `out_type == 'array'` won't need `['coef']` later
# TODO test that freq-averaged FOTS shape matches joint for out_type='array'

# set True to execute all test functions without pytest
run_without_pytest = 0


def test_alignment():
    """Ensure A.M. cosine's peaks are aligned across `psi2` joint slices,
    both spins, for `oversampling_fr='auto'`.
    """
    def max_row_idx(s):
        return np.argmax(np.sum(s**2, axis=-1))

    T = 2049
    J = 7
    Q = 16

    # generate A.M. cosine ###################################################
    f1, f2 = 8, 256
    t = np.linspace(0, 1, T, 1)
    a = (np.cos(2*np.pi * f1 * t) + 1) / 2
    c = np.cos(2*np.pi * f2 * t)
    x = a * c

    # scatter ################################################################
    for out_type in ('array', 'list'):
        scattering = TimeFrequencyScattering(
            J, T, Q, J_fr=4, Q_fr=2, average=True, oversampling=0,
            out_type=out_type, aligned=True, padtype='reflect')

        Scx = scattering(x)

        # assert peaks share an index ########################################
        meta = scattering.meta()
        S_all = {}
        for i, s in enumerate(Scx):
            n = meta['n'][i]
            if (n[1] == 0            # take earliest `sc_freq.psi1_f`
                    and n[0] >= 4):  # some `psi2` won't capture the peak
                S_all[i] = Scx[i]['coef']

        mx_idx = max_row_idx(list(S_all.values())[0])
        for i, s in S_all.items():
            mx_idx_i = max_row_idx(s)
            assert abs(mx_idx_i - mx_idx) < 2, (
                "{} != {} (Scx[{}], out_type={})").format(
                    mx_idx_i, mx_idx, i, out_type)


def test_shapes():
    """Ensure `out_type == 'array'` joint coeff slices have same shape."""
    T = 1024
    J = 6
    Q = 16

    x = np.random.randn(T)

    # scatter ################################################################
    for oversampling in (0, 1):
      for oversampling_fr in (0, 1):
        for aligned in (True, False):
          scattering = TimeFrequencyScattering(
              J, T, Q, J_fr=4, Q_fr=2, average=True, out_type='array',
              oversampling=oversampling, aligned=aligned)
          Scx = scattering(x)

          # assert slice shapes are equal ##############################
          meta = scattering.meta()
          S_all = {}
          for i, s in enumerate(Scx):
              if not np.isnan(meta['n'][i][1]):  # skip first-order
                  S_all[i] = s

          ref_shape = list(S_all.values())[0]['coef'].shape
          for i, s in S_all.items():
              assert s['coef'].shape == ref_shape, (
                  "{} != {} | (oversampling, oversampling_fr, aligned, n) = "
                  "({}, {}, {}, {})"
                  ).format(s['coef'].shape, ref_shape, oversampling,
                           oversampling_fr, aligned, tuple(meta['n'][i]))


def test_jtfs_vs_ts():
    """Test JTFS sensitivity to FDTS (frequency-dependent time shifts), and that
    time scattering is insensitive to it.
    """
    N = 2048
    f0 = N // 96
    n_partials = 5
    total_shift = N//16

    t = np.linspace(0, 1, N // 8, endpoint=False)
    window = scipy.signal.tukey(N//8, alpha=0.5)

    x = np.zeros(N)
    y = x.copy()
    for p in range(1, 1 + n_partials):
        tone = np.cos(2*np.pi * p*f0 * t)
        x_partial = tone * window
        x_partial = np.pad(x_partial, 7*N//16)
        partial_shift = int(total_shift * np.log2(p) / np.log2(n_partials)
                            ) - total_shift//2
        y_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        y += y_partial

    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = 16
    ts = Scattering1D(J=J, Q=Q, shape=N)
    jtfs = TimeFrequencyScattering(J=J, Q=Q, Q_fr=1, J_fr=4, shape=N,
                                   out_type="array")

    ts_x = ts(x)
    ts_y = ts(y)

    jtfs_x_list = jtfs(x)
    jtfs_y_list = jtfs(y)
    jtfs_x = np.concatenate([path["coef"] for path in jtfs_x_list])
    jtfs_y = np.concatenate([path["coef"] for path in jtfs_y_list])

    # get index of first joint coeff
    jmeta = jtfs.meta()
    first_joint_idx = [i for i, n in enumerate(jmeta['n'])
                       if not np.isnan(n[1])][0]
    arr_idx = sum(len(jtfs_x_list[i]['coef']) for i in range(len(jtfs_x_list))
                  if i < first_joint_idx)

    # skip zeroth-order
    ts_l1l2 = l1l2(ts_x[1:], ts_y[1:])
    # compare against joint coeffs only
    jtfs_l1l2 = l1l2(jtfs_x[arr_idx:], jtfs_y[arr_idx:])

    # max ratio limited by `N`; can do much better with longer input
    assert jtfs_l1l2 / ts_l1l2 > 3, "TS: %s\nJTFS: %s" % (ts_l1l2, jtfs_l1l2)
    assert ts_l1l2 < .1, "TS: %s" % ts_l1l2


def _l1l2(x):
    return np.sum(np.sqrt(np.mean(x**2, axis=1)), axis=0)

def l1l2(x0, x1):
    """Coeff distance measure; Thm 2.12 in https://arxiv.org/abs/1101.2286"""
    return _l1l2(x1 - x0) / _l1l2(x0)


if __name__ == '__main__':
    if run_without_pytest:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
    else:
        pytest.main([__file__, "-s"])
