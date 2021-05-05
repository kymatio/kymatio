import os
import pytest
import numpy as np
import scipy.signal
from kymatio.numpy import Scattering1D, TimeFrequencyScattering

# TODO no kymatio.numpy

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
        jtfs = TimeFrequencyScattering(
            J, T, Q, J_fr=4, Q_fr=2, average=True,
            out_type=out_type, aligned=True)

        Scx = jtfs(x)
        jmeta = jtfs.meta()

        # assert peaks share an index ########################################
        S_all = {}
        for pair in ('psi_t * psi_f_up', 'psi_t * psi_f_down'):
            S_all[pair] = {}
            for i, n in enumerate(jmeta['n'][pair]):
                if n[0] >= 4:  # some `psi2` won't capture peak
                    S_all[pair][i] = (Scx[pair][i]['coef'] if out_type == "list"
                                      else Scx[pair][i])

        first_coef = list(list(S_all.values())[0].values())[0]
        mx_idx = max_row_idx(first_coef)
        for pair in S_all:
            for i, s in S_all[pair].items():
                mx_idx_i = max_row_idx(s)
                assert abs(mx_idx_i - mx_idx) < 2, (
                    "{} != {} (Scx[{}][{}], out_type={})").format(
                        mx_idx_i, mx_idx, pair, i, out_type)


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
          jtfs = TimeFrequencyScattering(
              J, T, Q, J_fr=4, Q_fr=2, average=True, out_type='array',
              oversampling=oversampling, aligned=aligned)
          Scx = jtfs(x)
          jmeta = jtfs.meta()

          # assert slice shapes are equal ####################################
          # namely, # of freq rows and time shifts is same across pairs
          ref_shape = Scx['psi_t * psi_f_up'][0].shape
          for pair in Scx:
              if pair not in ('S0', 'S1'):
                  for i, s in enumerate(Scx[pair]):
                      assert s.shape == ref_shape, (
                          "{} != {} | (oversampling, oversampling_fr, aligned, n)"
                          " = ({}, {}, {}, {})").format(
                              s.shape, ref_shape, oversampling, oversampling_fr,
                              aligned, tuple(jmeta['n'][pair][i]))

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
    ts = Scattering1D(J=J, Q=Q, shape=N, pad_mode="zero", max_pad_factor=1)
    jtfs = TimeFrequencyScattering(J=J, Q=Q, Q_fr=1, J_fr=4, shape=N,
                                   out_type="array", max_pad_factor=1)

    # scatter
    ts_x  = ts(x)
    ts_xs = ts(xs)

    jtfs_x_all  = jtfs(x)
    jtfs_xs_all = jtfs(xs)
    jtfs_x  = concat_joint(jtfs_x_all)
    jtfs_xs = concat_joint(jtfs_xs_all)  # compare against joint coeffs only

    l2_ts = l2(ts_x, ts_xs)
    l2_jtfs = l2(jtfs_x, jtfs_xs)

    # max ratio limited by `N`; can do much better with longer input
    # and by comparing only against up & down
    assert l2_jtfs / l2_ts > 14, "\nTS: %s\nJTFS: %s" % (l2_ts, l2_jtfs)
    assert l2_ts < .006, "TS: %s" % l2_ts


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
    J_fr = 4
    F_all = [2**(J_fr), 2**(J_fr + 1)]
    th_all = [.19, .14]

    for th, F in zip(th_all, F_all):
        jtfs = TimeFrequencyScattering(J=J, Q=16, Q_fr=1, J_fr=J_fr, shape=N,
                                       F=F, out_type="array")
        # scatter
        jtfs_x0_all = jtfs(x0)
        jtfs_x1_all = jtfs(x1)
        jtfs_x0 = concat_joint(jtfs_x0_all)  # compare against joint coeffs only
        jtfs_x1 = concat_joint(jtfs_x1_all)

        l2_x0x1 = l2(jtfs_x0, jtfs_x1)

        # TODO is this value reasonable? it's much greater with different f0
        # (but same relative f1)
        assert l2_x0x1 < th, "{} > {} (F={})".format(l2_x0x1, th, F)


def test_up_vs_down():
    """Test that echirp yields significant disparity in up vs down coeffs."""
    N = 2048
    x = echirp(N)

    jtfs = TimeFrequencyScattering(shape=N, J=10, Q=16, J_fr=4, Q_fr=1)
    Scx = jtfs(x)

    E_up   = energy(Scx['psi_t * psi_f_up'])
    E_down = energy(Scx['psi_t * psi_f_down'])
    assert E_up / E_down > 17  # TODO reverse ratio after up/down fix


def test_meta():
    """Test that `TimeFrequencyScattering.meta()` matches output's meta."""
    def assert_equal(Scx, meta, field, pair, i):
        a, b = Scx[pair][i][field], meta[field][pair][i]
        errmsg = "(out[{0}][{1}][{2}], meta[{2}][{0}][{1}]) = ({3}, {4})".format(
            pair, i, field, a, b)
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

    Scx = jtfs(x)
    meta = jtfs.meta()

    for field in ('j', 'n', 's'):
        for pair in meta[field]:
            for i in range(len(meta[field][pair])):
                assert_equal(Scx, meta, field, pair, i)


def test_output():
    """Applies JTFS on a stored signal to make sure its output agrees with
    a previously calculated version. Tests for:
        0. (aligned, out_type, average_fr) = (True,  "list",  True)
        1. (aligned, out_type, average_fr) = (True,  "array", True)
        2. (aligned, out_type, average_fr) = (False, "array", True)
        3. (aligned, out_type, average_fr) = (True,  "list",  "global")
        4. [2.] + (resample_psi_fr, resample_phi_fr) = (False, False)
        5. special: params such that `sc_freq.J_pad_fo > sc_freq.J_pad_max`
            - i.e. all first-order coeffs pad to greater than longest set of
            second-order, as in `U1 * phi_t * phi_f` and
            `(U1 * phi_t * psi_f) * phi_t * phi_f`.
    """
    def _load_data(test_num, test_data_dir):
        """Also see data['code']."""
        def is_coef(k):
            return ':' in k and k.split(':')[-1].isdigit()
        def not_param(k):
            return k in ('code', 'x') or is_coef(k)

        data = np.load(os.path.join(test_data_dir, f'test_jtfs_{test_num}.npz'))
        x = data['x']
        out_stored = [data[k] for k in data.files if is_coef(k)]
        out_stored_keys = [k for k in data.files if is_coef(k)]

        params = {}
        for k in data.files:
            if not_param(k):
                continue

            if k in ('average', 'aligned', 'resample_psi_fr', 'resample_phi_fr'):
                params[k] = bool(data[k])
            elif k == 'average_fr':
                params[k] = (str(data[k]) if str(data[k]) == 'global' else
                             bool(data[k]))
            elif k == 'out_type':
                params[k] = str(data[k])
            else:
                params[k] = int(data[k])

        params_str = "Test #%s:\n" % test_num
        for k, v in params.items():
            params_str += "{}={}\n".format(k, str(v))
        return x, out_stored, out_stored_keys, params, params_str

    test_data_dir = os.path.dirname(__file__)
    num_tests = sum("test_jtfs_" in p for p in os.listdir(test_data_dir))

    for test_num in range(num_tests):
        (x, out_stored, out_stored_keys, params, params_str
         ) = _load_data(test_num, test_data_dir)

        jtfs = TimeFrequencyScattering(**params, max_pad_factor=1)
        out = jtfs(x)

        n_coef_out = sum(1 for pair in out for c in out[pair])
        n_coef_out_stored = len(out_stored)
        assert n_coef_out == n_coef_out_stored, (
            "out vs stored number of coeffs mismatch ({} != {})\n{}"
            ).format(n_coef_out, n_coef_out_stored, params_str)

        i_s = 0
        for pair in out:
            for i, o in enumerate(out[pair]):
                o = o if params['out_type'] == 'array' else o['coef']
                o_stored, o_stored_key = out_stored[i_s], out_stored_keys[i_s]
                assert o.shape == o_stored.shape, (
                    "out[{0}][{1}].shape != out_stored[{2}].shape "
                    "({3} != {4})\n{5}".format(pair, i, o_stored_key, o.shape,
                                               o_stored.shape, params_str))
                assert np.allclose(o, o_stored), (
                    "out[{0}][{1}] != out_stored[{2}] (MAE={3:.5f})\n{4}"
                    ).format(pair, i, o_stored_key, np.abs(o - o_stored).mean(),
                             params_str)
                i_s += 1

### helper methods ###########################################################
# TODO move to (and create) tests/utils.py?
def _l2(x):
    return np.sqrt(np.sum(np.abs(x)**2))

def l2(x0, x1):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    return _l2(x1 - x0) / _l2(x0)

def energy(x):
    return np.sum(np.abs(x)**2)

# def _l1l2(x):
#     return np.sum(np.sqrt(np.sum(np.abs(x)**2, axis=1)), axis=0)

# def l1l2(x0, x1):
#     """Coeff distance measure; Thm 2.12 in https://arxiv.org/abs/1101.2286"""
#     return _l2(x1 - x0) / _l2(x0)

def concat_joint(Scx, out_type="array"):
    return np.vstack([(c if out_type == "array" else c['coef'])
                      for pair, c in Scx.items()
                      if pair not in ('S0', 'S1')])


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


def echirp(N, fmin=.1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)

    phi = 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return np.cos(phi)


if __name__ == '__main__':
    if run_without_pytest:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_up_vs_down()
        test_meta()
        test_output()
    else:
        pytest.main([__file__, "-s"])
