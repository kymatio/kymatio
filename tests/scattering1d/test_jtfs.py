import pytest
import numpy as np
import scipy.signal
from kymatio.numpy import Scattering1D, TimeFrequencyScattering

# TODO no kymatio.numpy
# TODO `out_type == 'array'` won't need `['coef']` later
# TODO test that freq-averaged FOTS shape matches joint for out_type='array'
# TODO joint coeffs exclude freq-averaged U1 (fix after sorting meta stuff)
# TODO phase-shift sensitivity

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
            out_type=out_type, aligned=True)

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
    # design signal
    N = 2048
    f0 = N // 12
    n_partials = 5
    total_shift = N//12
    seg_len = N//8

    x, xs = fdts(N, n_partials, total_shift, f0, seg_len)

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = 16
    ts = Scattering1D(J=J, Q=Q, shape=N)
    jtfs = TimeFrequencyScattering(J=J, Q=Q, Q_fr=1, J_fr=4, shape=N,
                                   out_type="array")

    # scatter
    ts_x  = ts(x)
    ts_xs = ts(xs)

    jtfs_x_list  = jtfs(x)
    jtfs_xs_list = jtfs(xs)
    jtfs_x  = np.concatenate([path["coef"] for path in jtfs_x_list])
    jtfs_xs = np.concatenate([path["coef"] for path in jtfs_xs_list])

    # get index of first joint coeff
    jmeta = jtfs.meta()
    first_joint_idx = [i for i, n in enumerate(jmeta['n'])
                       if not np.isnan(n[1])][0]
    arr_idx = sum(len(jtfs_x_list[i]['coef']) for i in range(len(jtfs_x_list))
                  if i < first_joint_idx)

    # skip zeroth-order
    l2_ts = l2(ts_x[1:], ts_xs[1:])
    # compare against joint coeffs only
    l2_jtfs = l2(jtfs_x[arr_idx:], jtfs_xs[arr_idx:])

    # max ratio limited by `N`; can do much better with longer input
    assert l2_jtfs / l2_ts > 3, "\nTS: %s\nJTFS: %s" % (l2_ts, l2_jtfs)
    assert l2_ts < .1, "TS: %s" % l2_ts


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
    Q = 16
    jtfs = TimeFrequencyScattering(J=J, Q=Q, Q_fr=1, J_fr=4, shape=N,
                                   out_type="array")

    # scatter
    jtfs_x0_list = jtfs(x0)
    jtfs_x1_list = jtfs(x1)
    jtfs_x0 = np.concatenate([path["coef"] for path in jtfs_x0_list])
    jtfs_x1 = np.concatenate([path["coef"] for path in jtfs_x1_list])

    # get index of first joint coeff
    jmeta = jtfs.meta()
    first_joint_idx = [i for i, n in enumerate(jmeta['n'])
                       if not np.isnan(n[1])][0]
    arr_idx = sum(len(jtfs_x0_list[i]['coef']) for i in range(len(jtfs_x0_list))
                  if i < first_joint_idx)

    # compare against joint coeffs only
    l2_x0x1 = l2(jtfs_x0[arr_idx:], jtfs_x1[arr_idx:])

    # TODO is this value reasonable? it's much greater with different f0
    # (but same relative f1)
    th = .20
    assert l2_x0x1 < th, "{} > {}".format(l2_x0x1, th)


def test_meta():
    """Test that `TimeFrequencyScattering.meta()` matches output's meta."""
    def assert_equal(a, b, field, i):
        errmsg = "{}: (out[{}], meta[{}]) = ({}, {})".format(field, i, i, a, b)
        if len(a) == len(b):
            assert np.all(a == b), errmsg
        elif len(a) == 0:
            assert np.all(np.isnan(b)), errmsg
        elif len(a) < len(b):
            assert a[0] == b[0], errmsg
            assert np.isnan(b[1]), errmsg

    N = 2048
    x = np.random.randn(N)

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = 16
    jtfs = TimeFrequencyScattering(J=J, Q=Q, Q_fr=1, shape=N, out_type="list")

    out = jtfs(x)
    meta = jtfs.meta()

    for field in ('j', 'n', 's'):
        for i in range(len(meta[field])):
            assert_equal(out[i][field], meta[field][i], field, i)

### helper methods ###########################################################
# TODO move to (and create) tests/utils.py?
def _l2(x):
    return np.linalg.norm(x)

def l2(x0, x1):
    """Coeff distance measure; Eq 2.25 in https://arxiv.org/abs/1101.2286"""
    return _l2(x1 - x0) / _l2(x0)


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=False)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    window = np.pad(window, (N - len(window)) // 2)

    x = np.zeros(N)
    xs = x.copy()
    for p in range(1, 1 + n_partials):
        x_partial = np.sin(2*np.pi * p*f0 * t) * window
        partial_shift = int(total_shift * np.log2(p) / np.log2(n_partials)
                            ) - total_shift//2
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial
    return x, xs


if __name__ == '__main__':
    if run_without_pytest:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_meta()
    else:
        pytest.main([__file__, "-s"])
