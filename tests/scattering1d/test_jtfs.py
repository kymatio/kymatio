import pytest
import numpy as np
import scipy.signal
import warnings
from pathlib import Path
from kymatio import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import drop_batch_dim_jtfs

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to print assertion errors rather than raising them in `test_output()`
output_test_print_mode = 1


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
      for resample_psi_fr in (True, 'exclude'):
        if resample_psi_fr == 'exclude' and out_3D:
            continue  # incompatible
        for J_fr in (3, 5):
          out_type = ('dict:array' if out_3D else
                      'dict:list')  # for convenience
          test_params = dict(out_3D=out_3D,
                             resample_filters_fr=(resample_psi_fr, True))
          test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                      test_params.items())
          jtfs = TimeFrequencyScattering1D(
              J, N, Q, J_fr=J_fr, Q_fr=Q_fr, F=F, average=True, average_fr=True,
              aligned=True, out_type=out_type, frontend=default_backend,
              **test_params)

          Scx = jtfs(x)
          Scx = drop_batch_dim_jtfs(Scx)

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
              # assert not all J_pad_fr are same so test covers this case
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
    f0 = N // 12
    n_partials = 5
    total_shift = N//12
    seg_len = N//8

    x, xs = fdts(N, n_partials, total_shift, f0, seg_len)

    # make scattering objects
    J = int(np.log2(N) - 1)  # have 2 time units at output
    Q = (16, 2)
    kw = dict(J=J, shape=N, max_pad_factor=1, frontend=default_backend)
    ts = Scattering1D(Q=Q[0], pad_mode="zero", out_type='array', **kw)
    jtfs = TimeFrequencyScattering1D(Q=Q, Q_fr=2, J_fr=4, average_fr=True,
                                     out_3D=True, out_type='dict:array', **kw)

    # scatter
    ts_x  = ts(x)
    ts_xs = ts(xs)

    jtfs_x_all  = jtfs(x)
    jtfs_xs_all = jtfs(xs)
    jtfs_x  = concat_joint(jtfs_x_all)
    jtfs_xs = concat_joint(jtfs_xs_all)  # compare against joint coeffs only

    l2_ts = l2(ts_x, ts_xs)
    l2_jtfs = l2(jtfs_x, jtfs_xs)

    # max ratio limited by `N`; can do better with longer input
    # and by comparing only against up & down
    assert l2_jtfs / l2_ts > 25, "\nTS: %s\nJTFS: %s" % (l2_ts, l2_jtfs)
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
    th_all = [.21, .14]

    for th, F in zip(th_all, F_all):
        jtfs = TimeFrequencyScattering1D(J=J, Q=16, Q_fr=1, J_fr=J_fr, shape=N,
                                         F=F, average_fr=True, out_3D=True,
                                         out_type='dict:array',
                                         frontend=default_backend)
        # scatter
        jtfs_x0_all = jtfs(x0)
        jtfs_x1_all = jtfs(x1)
        jtfs_x0 = concat_joint(jtfs_x0_all)  # compare against joint coeffs only
        jtfs_x1 = concat_joint(jtfs_x1_all)

        l2_x0x1 = l2(jtfs_x0, jtfs_x1)

        assert l2_x0x1 < th, "{} > {} (F={})".format(l2_x0x1, th, F)


def test_up_vs_down():
    """Test that echirp yields significant disparity in up vs down coeffs."""
    N = 2048
    x = echirp(N)

    jtfs = TimeFrequencyScattering1D(shape=N, J=7, Q=8, J_fr=4, Q_fr=1,
                                     average_fr=True, out_3D=True,
                                     out_type='dict:array',
                                     frontend=default_backend)
    Scx = jtfs(x)

    E_up   = energy(Scx['psi_t * psi_f_up'])
    E_down = energy(Scx['psi_t * psi_f_down'])
    assert E_down / E_up > 125


def test_exclude():
    """Test that outputs of `resample_psi_fr=='exclude'` are a subset of
    `==True` (i.e. equal wherever both exist).
    """
    N = 1024
    x = echirp(N)

    params = dict(shape=N, J=9, Q=8, J_fr=3, Q_fr=1, average_fr=True,
                  out_type='dict:list', frontend=default_backend)
    test_params_str = '\n'.join(f'{k}={v}' for k, v in params.items())
    jtfs0 = TimeFrequencyScattering1D(**params,
                                      resample_filters_fr=(True, True))
    jtfs1 = TimeFrequencyScattering1D(**params,
                                      resample_filters_fr=('exclude', True))

    # required otherwise 'exclude' == True
    assert_pad_difference(jtfs0, test_params_str)
    # reproduce case with different J_pad_fr
    assert jtfs0.J_pad_fr != jtfs1.J_pad_fr

    Scx0 = jtfs0(x)
    Scx1 = jtfs1(x)

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

            if n0 != n1:
                # check 1's pad as indexed by 0, since n0 lags n1 and might
                # have e.g. pad1[n0=5]==(max-1), pad[n1=6]==max, but we're still
                # iterating n==5 so appropriate comparison is at 5
                pad, pad_max = jtfs1.J_pad_fr[n0[0]], jtfs1.J_pad_fr_max
                assert pad != pad_max, (
                    "{} == {} | {}\n(must have sub-maximal `J_pad_fr` for "
                    "mismatched `n`)").format(pad, pad_max, info)
                continue

            assert c0.shape == c1.shape, ("shape mismatch: {} != {} | {}".format(
                c0.shape, c1.shape, info))
            ae = np.abs(c0 - c1)
            # lower threshold because pad differences are possible
            assert np.allclose(c0, c1, atol=5e-7), (
                "{} | MeanAE={:.2e}, MaxAE={:.2e}"
                ).format(info, ae.mean(), ae.max())
            i1 += 1


def test_max_pad_factor_fr():
    """Test that low `max_pad_factor_fr` works with high `F`."""
    N = 2048
    x = echirp(N)

    for aligned in (True, False):
        for resample_filters_fr in (True, False):
            jtfs = TimeFrequencyScattering1D(
                shape=N, J=10, Q=20, J_fr=4, Q_fr=1, F=256, average_fr=True,
                max_pad_factor_fr=1, aligned=aligned, out_3D=True,
                resample_filters_fr=resample_filters_fr, frontend=default_backend)

            params_str = "aligned={}, resample_filters_fr={}".format(
                aligned, resample_filters_fr)
            try:
                _ = jtfs(x)
            except Exception as e:
                print("Failed on %s with" % params_str)
                raise e


def test_no_second_order_filters():
    """Reproduce edge case: configuration yields no second-order wavelets
    so can't do JTFS.
    """
    with pytest.raises(ValueError) as record:
        _ = TimeFrequencyScattering1D(shape=512, J=1, Q=1,
                                      frontend=default_backend)
        assert "no second-order filters" in record.value.args[0]


def test_backends():
    for backend in ('tensorflow', 'torch'):
        if cant_import(backend):
            continue
        elif backend == 'torch':
            import torch
        elif backend == 'tensorflow':
            import tensorflow as tf

        N = 2048
        x = echirp(N)
        x = np.vstack([x, x, x])
        x = (tf.constant(x) if backend == 'tensorflow' else
             torch.from_numpy(x))

        jtfs = TimeFrequencyScattering1D(shape=N, J=8, Q=8, J_fr=3, Q_fr=1,
                                         average_fr=True, out_3D=True,
                                         out_type='dict:array', frontend=backend)
        Scx = jtfs(x)

        E_up   = energy(Scx['psi_t * psi_f_up'])
        E_down = energy(Scx['psi_t * psi_f_down'])
        assert E_down / E_up > 80


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
    T = 2**12
    for device in devices:
        jtfs = TimeFrequencyScattering1D(J, T, Q, frontend='torch',
                                         max_pad_factor=1).to(device)
        x = torch.randn(2, T, requires_grad=True, device=device)

        s = jtfs.forward(x)
        loss = torch.sum(torch.abs(s))
        loss.backward()
        assert torch.max(torch.abs(x.grad)) > 0.


def test_meta():
    """Tests that meta values and structures match those of output for all
    combinations of
        - out_3D (True only with average_fr=True and resample_psi_fr != 'exclude')
        - average_fr
        - aligned
        - resample_psi_fr
        - resample_phi_fr
    a total of 32 tests. All possible ways of packing the same coefficients
    (via `out_type`) aren't tested.

    Not tested:
        - average
        - average_fr_global
        - oversampling_fr
        - max_padding_fr
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
                            test_params_str, jtfs):
        """Assert that non-NaN values are equal."""
        a, b = Scx[pair][i][field], jmeta[field][pair][meta_idx[0]]
        errmsg = ("(out[{0}][{1}][{2}], jmeta[{2}][{0}][{3}]) = ({4}, {5})\n{6}"
                  ).format(pair, i, field, meta_idx[0], a, b, test_params_str)

        meta_len = b.shape[-1]
        if field != 's':
            assert meta_len == 3, ("all meta fields (except spin) must pad to "
                                 "length 3: %s" % errmsg)
            assert len(a) > 0, ("all computed metas (except spin) must append "
                                "something: %s" % errmsg)

        if field == 's' and pair in ('S0', 'S1'):
            assert len(a) == 0 and np.isnan(b), errmsg
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

    def run_test(params, test_params):
        jtfs = TimeFrequencyScattering1D(**params, **test_params,
                                         frontend=default_backend)
        test_params_str = '\n'.join(f'{k}={v}' for k, v in test_params.items())

        resample_psi_fr = test_params['resample_filters_fr'][0]
        if resample_psi_fr in (False, 'exclude'):
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
        for field in ('j', 'n', 's'):
          for pair in jmeta[field]:
            meta_idx = [0]
            assert_equal_lengths(Scx, jmeta, field, pair, out_3D,
                                 test_params_str, jtfs)
            for i in range(len(Scx[pair])):
                assert_equal_values(Scx, jmeta, field, pair, i, meta_idx,
                                    out_3D, test_params_str, jtfs)

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
        if out_3D and not average_fr:
            continue  # invalid option
        for aligned in (True, False):
          for resample_psi_fr in (True, False):
            for resample_phi_fr in (True, False):
                test_params = dict(
                    out_3D=out_3D, average_fr=average_fr, aligned=aligned,
                    resample_filters_fr=(resample_psi_fr, resample_phi_fr))
                run_test(params, test_params)

    # reproduce this case separately; above doesn't test where 'exclude' fails
    N = 1024
    x = np.random.randn(N)
    J = int(np.log2(N) - 1)
    J_fr = 3
    params = dict(shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type)

    resample_psi_fr = 'exclude'
    for average_fr in (True, False):
      for aligned in (True, False):
        for resample_phi_fr in (True, False):
            test_params = dict(
                out_3D=False, average_fr=average_fr, aligned=aligned,
                resample_filters_fr=(resample_psi_fr, resample_phi_fr))
            run_test(params, test_params)


def test_output():
    """Applies JTFS on a stored signal to make sure its output agrees with
    a previously calculated version. Tests for:

          (aligned, average_fr, out_3D, out_type,     F)
        0. True     True        False   'dict:list'   8
        1. True     True        True    'dict:array'  8  # TODO make this test 0
        2. False    True        True    'dict:array'  8
        3. True     True        False   'dict:list'   'global'
        4. True     False       False   'dict:array'  8

        5. [2.] + (resample_psi_fr, resample_phi_fr) = (False, False)
        6. special: params such that `sc_freq.J_pad_fo > sc_freq.J_pad_max`
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
            elif k == 'resample_filters_fr':
                params[k] = (bool(data[k]) if len(data[k]) == 1 else
                             tuple(data[k]))
            elif k == 'F':
                params[k] = (str(data[k]) if str(data[k]) == 'global' else
                             int(data[k]))
            elif k == 'out_type':
                params[k] = str(data[k])
            else:
                params[k] = int(data[k])

        params_str = "Test #%s:\n" % test_num
        for k, v in params.items():
            params_str += "{}={}\n".format(k, str(v))
        return x, out_stored, out_stored_keys, params, params_str

    test_data_dir = Path(__file__).parent
    num_tests = sum((p.name.startswith('test_jtfs_') and p.suffix == '.npz')
                    for p in Path(test_data_dir).iterdir())

    for test_num in range(num_tests):
        (x, out_stored, out_stored_keys, params, params_str
         ) = _load_data(test_num, test_data_dir)
        jtfs = TimeFrequencyScattering1D(**params, frontend=default_backend)
        out = jtfs(x)

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
        for pair in out:
            for i, o in enumerate(out[pair]):
                # assert equal shapes
                o = o if params['out_type'] == 'dict:array' else o['coef']
                o_stored, o_stored_key = out_stored[i_s], out_stored_keys[i_s]
                errmsg = ("out[{0}][{1}].shape != out_stored[{2}].shape\n"
                          "({3} != {4})\n{5}"
                          ).format(pair, i, o_stored_key, o.shape, o_stored.shape,
                                   params_str)
                if output_test_print_mode and o.shape != o_stored.shape:
                    print(errmsg)
                    i_s += 1
                    continue
                else:
                    assert o.shape == o_stored.shape, errmsg

                # assert equal values
                adiff = np.abs(o - o_stored)
                errmsg = ("out[{0}][{1}] != out_stored[{2}]\n"
                          "(MeanAE={3:.2e}, MaxAE={4:.2e})\n{5}"
                          ).format(pair, i, o_stored_key, adiff.mean(),
                                   adiff.max(), params_str)
                if output_test_print_mode and not np.allclose(o, o_stored):
                    print(errmsg)
                else:
                    assert np.allclose(o, o_stored), errmsg
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
    if isinstance(x, np.ndarray):
        return np.sum(np.abs(x)**2)
    elif 'torch' in str(type(x)):
        import torch
        return torch.sum(torch.abs(x)**2)
    elif 'tensorflow' in str(type(x)):
        import tensorflow as tf
        return tf.reduce_sum(tf.abs(x)**2)

# def _l1l2(x):
#     return np.sum(np.sqrt(np.sum(np.abs(x)**2, axis=1)), axis=0)

# def l1l2(x0, x1):
#     """Coeff distance measure; Thm 2.12 in https://arxiv.org/abs/1101.2286"""
#     return _l2(x1 - x0) / _l2(x0)

def concat_joint(Scx, out_type="array"):
    Scx = drop_batch_dim_jtfs(Scx)
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


def assert_pad_difference(jtfs, test_params_str):
    assert not all(
        J_pad_fr == jtfs.J_pad_fr_max
        for J_pad_fr in jtfs.J_pad_fr if J_pad_fr != -1
        ), "{}\nJ_pad_fr={}\nshape_fr={}".format(
            test_params_str, jtfs.J_pad_fr, jtfs.shape_fr)


def cant_import(backend_name):
    if backend_name == 'numpy':
        return False
    elif backend_name == 'torch':
        try:
            import torch
        except ImportError:
            warnings.warn("Failed to import torch")
            return True
    elif backend_name == 'tensorflow':
        try:
            import tensorflow
        except ImportError:
            warnings.warn("Failed to import tensorflow")
            return True


if __name__ == '__main__':
    if run_without_pytest:
        test_alignment()
        test_shapes()
        test_jtfs_vs_ts()
        test_freq_tp_invar()
        test_up_vs_down()
        test_exclude()
        test_no_second_order_filters()
        test_max_pad_factor_fr()
        test_backends()
        test_differentiability_torch()
        test_meta()
        test_output()
    else:
        pytest.main([__file__, "-s"])
