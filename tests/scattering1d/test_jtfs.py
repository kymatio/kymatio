import pytest
import numpy as np
from pathlib import Path
from kymatio import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import drop_batch_dim_jtfs, coeff_energy, fdts, echirp, l2
from kymatio.visuals import coeff_distance_jtfs, compare_distances_jtfs
from utils import cant_import

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to print assertion errors rather than raising them in `test_output()`
output_test_print_mode = 1
# set True to print assertion values of certain tests
metric_verbose = 1


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
    assert l2_jtfs / l2_ts > 21, ("'nJTFS/TS: %s \nTS: %s\nJTFS: %s"
                                  )% (l2_jtfs / l2_ts, l2_ts, l2_jtfs)
    assert l2_ts < .006, "TS: %s" % l2_ts
    # TODO take l2 distance from stride-adjusted coeffs?
    # TODO also compare max per-coeff ratio

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
    J_fr = 6
    F_all = [32, 64]

    pair_distances, global_distances = [], []
    for F in F_all:
        jtfs = TimeFrequencyScattering1D(J=J, Q=16, Q_fr=1, J_fr=J_fr, shape=N,
                                         F=F, average_fr=True, out_3D=False,
                                         out_type='dict:array',
                                         oversampling=0, oversampling_fr=0,
                                         out_exclude=('S0', 'S1'),
                                         # pad_mode='zero', pad_mode_fr='zero',
                                         pad_mode='reflect',
                                         pad_mode_fr='conj-reflect-zero',
                                         frontend=default_backend)
        # scatter
        jtfs_x0_all = jtfs(x0)
        jtfs_x1_all = jtfs(x1)

        # compute & append distances
        _, pair_dist = coeff_distance_jtfs(jtfs_x0_all, jtfs_x1_all,
                                           jtfs.meta(), plots=False)
        pair_distances.append(pair_dist)

        jtfs_x0 = concat_joint(jtfs_x0_all)
        jtfs_x1 = concat_joint(jtfs_x1_all)
        global_distances.append(l2(jtfs_x0, jtfs_x1))

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
    # TODO 'zero' & 'zero' pads attain 98.9
    # TODO include both in testing (zero & reflect)?
    N = 2048
    x = echirp(N)

    jtfs = TimeFrequencyScattering1D(shape=N, J=7, Q=8, J_fr=4, F=4, Q_fr=2,
                                     average_fr=True, out_type='dict:array',
                                     pad_mode='reflect',
                                     pad_mode_fr='conj-reflect-zero',
                                     frontend=default_backend)
    Scx = jtfs(x)
    jmeta = jtfs.meta()

    E_up   = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_up')
    E_down = coeff_energy(Scx, jmeta, pair='psi_t * psi_f_down')
    th = 81
    assert E_down / E_up > th, "{} < {}".format(E_down / E_up, th)

    if metric_verbose:
        print(("\nFDTS directional sensitivity:\n"
               "E_down/E_up = {:.1f}\n").format(E_down / E_up))


def test_sampling_psi_fr_exclude():
    """Test that outputs of `sampling_psi_fr=='exclude'` are a subset of
    `==True` (i.e. equal wherever both exist).
    """
    N = 1024
    x = echirp(N)

    params = dict(shape=N, J=9, Q=8, J_fr=3, Q_fr=2, average_fr=True,
                  out_type='dict:list', frontend=default_backend)
    test_params_str = '\n'.join(f'{k}={v}' for k, v in params.items())
    jtfs0 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('resample', 'resample'))
    jtfs1 = TimeFrequencyScattering1D(
        **params, sampling_filters_fr=('exclude', 'resample'))

    # required otherwise 'exclude' == 'resample'
    assert_pad_difference(jtfs0, test_params_str)
    # reproduce case with different J_pad_fr
    assert jtfs0.J_pad_fr != jtfs1.J_pad_fr, jtfs0.J_pad_fr

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
                # Mismatched `n` should only happen for mismatched `pad_fr`.
                # Check 1's pad as indexed by 0, since n0 lags n1 and might
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
    """Test that low and variable `max_pad_factor_fr` works."""
    N = 1024
    x = echirp(N)

    for aligned in (True, False)[1:]:
        for sampling_filters_fr in ('resample', 'exclude', 'recalibrate'):
          for max_pad_factor_fr in (0, 1, [1, 2, 0, 1]):
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
    """Test that `T==N` and `F==pow2(shape_fr_max)` doesn't error, and outputs
    close to `T==N-1` and `F==pow2(shape_fr_max)-1`
    """
    np.random.seed(0)
    N = 512
    params = dict(shape=N, J=9, Q=4, J_fr=5, Q_fr=2, average=True,
                  average_fr=True, out_type='dict:array', pad_mode='reflect',
                  pad_mode_fr='conj-reflect-zero', max_pad_factor=None,
                  max_pad_factor_fr=None, frontend=default_backend)
    x = echirp(N)
    x += np.random.randn(N)

    outs = {}
    metas = {}
    Ts, Fs = (N - 1, N), (2**5 - 1, 2**5)
    for T in Ts:
        # shape_fr_max ~= Q*max(p2['j'] for p2 in psi2_f); found 29 at runtime
        for F in Fs:
            jtfs = TimeFrequencyScattering1D(**params, T=T, F=F)
            outs[ (T, F)] = jtfs(x)
            metas[(T, F)] = jtfs.meta()
            # print(T, F, '--',
            #       *[getattr(jtfs.sc_freq, k) for k in
            #         ('J_pad_fr_max', 'min_to_pad_fr_max', '_pad_fr_phi',
            #          '_pad_fr_psi')])  # TODO

    T0F0 = coeff_energy(outs[(Ts[0], Fs[0])], metas[(Ts[0], Fs[0])])
    T0F1 = coeff_energy(outs[(Ts[0], Fs[1])], metas[(Ts[0], Fs[1])])
    T1F0 = coeff_energy(outs[(Ts[1], Fs[0])], metas[(Ts[1], Fs[0])])
    T1F1 = coeff_energy(outs[(Ts[1], Fs[1])], metas[(Ts[1], Fs[1])])

    if metric_verbose:
        print("\nGlobal averaging reldiffs:")

    th = .1
    for pair in T0F0:
        ref = T0F0[pair]
        reldiff01 = abs(T0F1[pair] - ref) / ref
        reldiff10 = abs(T1F0[pair] - ref) / ref
        reldiff11 = abs(T1F1[pair] - ref) / ref
        assert reldiff01 < th, "%s > %s" % (reldiff01, th)
        assert reldiff10 < th, "%s > %s" % (reldiff10, th)
        assert reldiff11 < th, "%s > %s" % (reldiff11, th)

        if metric_verbose:
            print("(01, 10, 11) = ({:.2e}, {:.2e}, {:.2e}) | {}".format(
                reldiff01, reldiff10, reldiff11, pair))


def test_no_second_order_filters():
    """Reproduce edge case: configuration yields no second-order wavelets
    so can't do JTFS.
    """
    with pytest.raises(ValueError) as record:
        _ = TimeFrequencyScattering1D(shape=512, J=1, Q=1,
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

        # pick one sample
        out = {}
        for pair, coef in Scx.items():
            out[pair] = coef[0]

        E_up   = coeff_energy(out, jmeta, pair='psi_t * psi_f_up')
        E_down = coeff_energy(out, jmeta, pair='psi_t * psi_f_down')
        th = 43
        assert E_down / E_up > th, "{:.2f} < {}".format(E_down / E_up, th)


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
    Q = 6
    N = 512
    n_iters = 22
    jtfs = TimeFrequencyScattering1D(J, N, Q, frontend='torch', out_type='array',
                                     max_pad_factor=1, max_pad_factor_fr=2
                                     ).to(device)

    y = torch.from_numpy(echirp(N).astype('float32')).to(device)
    Sy = jtfs(y)

    torch.manual_seed(0)
    x = torch.randn(N, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([x], lr=.4)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for i in range(n_iters):
        optimizer.zero_grad()
        Sx = jtfs(x)
        loss = loss_fn(Sx, Sy)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().numpy()))

    th = 1e-5
    end_ratio = losses[0] / losses[-1]
    assert end_ratio > 25, end_ratio
    assert min(losses) < th, "{:.2e} > {}".format(min(losses), th)

    if metric_verbose:
        print(("\nReconstruction (torch):\n(end_start_ratio, min_loss) = "
               "({:.1f}, {:.2e})").format(end_ratio, min(losses)))


def test_meta():
    """Tests that meta values and structures match those of output for all
    combinations of
        - out_3D (True only with average_fr=True and sampling_psi_fr != 'exclude')
        - average_fr
        - average
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
    a total of 56 tests. All possible ways of packing the same coefficients
    (via `out_type`) aren't tested.

    Not tested:
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

    def run_test(params, test_params):
        jtfs = TimeFrequencyScattering1D(**params, **test_params,
                                         frontend=default_backend)
        test_params_str = '\n'.join(f'{k}={v}' for k, v in test_params.items())

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
            for sampling_psi_fr in ('resample', 'recalibrate'):
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
    for average_fr in (True, False):
      for average in (True, False):
        for aligned in (True, False):
          for sampling_phi_fr in ('resample', 'recalibrate'):
              test_params = dict(
                  out_3D=False, average_fr=average_fr, average=average,
                  aligned=aligned,
                  sampling_filters_fr=(sampling_psi_fr, sampling_phi_fr))
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

        5. [2.] + `sampling_psi_fr = sampling_phi_fr = 'recalibrate'`
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
        mean_aes, max_aes = [0], [0]
        already_printed_test_info, max_mean_info, max_max_info = False, None, None
        for pair in out:
            for i, o in enumerate(out[pair]):
                # assert equal shapes
                o = o if params['out_type'] == 'dict:array' else o['coef']
                o_stored, o_stored_key = out_stored[i_s], out_stored_keys[i_s]
                errmsg = ("out[{}][{}].shape != out_stored[{}].shape\n"
                          "({} != {})\n"
                          ).format(pair, i, o_stored_key, o.shape, o_stored.shape)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and o.shape != o_stored.shape:
                    print(errmsg)
                    already_printed_test_info = True
                    i_s += 1
                    continue
                else:
                    assert o.shape == o_stored.shape, errmsg

                # store info for printing
                adiff = np.abs(o - o_stored)
                mean_ae, max_ae = adiff.mean(), adiff.max()
                if mean_ae > max(mean_aes):
                    max_mean_info = "out[%s][%s]" % (pair, i)
                if max_ae > max(max_aes):
                    max_max_info  = "out[%s][%s]" % (pair, i)
                mean_aes.append(mean_ae)
                max_aes.append(max_ae)

                # assert equal values
                errmsg = ("out[{}][{}] != out_stored[{}]\n"
                          "(MeanAE={:.2e}, MaxAE={:.2e})\n"
                          ).format(pair, i, o_stored_key,
                                   mean_aes[-1], max_aes[-1],)
                if not already_printed_test_info:
                    errmsg += params_str

                if output_test_print_mode and not np.allclose(o, o_stored):
                    print(errmsg)
                    already_printed_test_info = True
                else:
                    assert np.allclose(o, o_stored), errmsg
                i_s += 1

        if output_test_print_mode:
            if max_mean_info is not None:
                print("// max_meanAE = {:.2e} | {}\n".format(max(mean_aes),
                                                             max_mean_info))
            if max_max_info is not None:
                print("// max_maxAE = {:.2e} | {}\n".format(max(max_aes),
                                                            max_max_info))

### helper methods ###########################################################
def energy(x):
    if isinstance(x, np.ndarray):
        return np.sum(np.abs(x)**2)
    elif 'torch' in str(type(x)):
        import torch
        return torch.sum(torch.abs(x)**2)
    elif 'tensorflow' in str(type(x)):
        import tensorflow as tf
        return tf.reduce_sum(tf.abs(x)**2)

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
        J_pad_fr == jtfs.J_pad_fr_max
        for J_pad_fr in jtfs.J_pad_fr if J_pad_fr != -1
        ), "\n{}\nJ_pad_fr={}\nshape_fr={}".format(
            test_params_str, jtfs.J_pad_fr, jtfs.shape_fr)


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
        test_backends()
        test_differentiability_torch()
        test_reconstruction_torch()
        test_meta()
        test_output()
    else:
        pytest.main([__file__, "-s"])
