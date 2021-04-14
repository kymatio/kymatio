import pytest
import numpy as np
from kymatio.numpy import TimeFrequencyScattering

# TODO no kymatio.numpy
# TODO `out_type == 'array'` won't need `['coef']` later

# set True to execute all test functions without pytest
run_without_pytest = 1


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
            out_type=out_type, oversampling_fr='auto')

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
            assert mx_idx_i == mx_idx, "{} != {} (Scx[{}], out_type={})".format(
                mx_idx_i, mx_idx, i, out_type)


def test_shapes():
    """Ensure `out_type == 'array'` joint coeff slices have same shape."""
    T = 2049
    J = 7
    Q = 16

    x = np.random.randn(T)

    # scatter ################################################################
    scattering = TimeFrequencyScattering(J, T, Q, J_fr=4, Q_fr=2, average=True,
                                         oversampling=0, out_type='array',
                                         oversampling_fr='auto')
    Scx = scattering(x)

    # assert peaks share an index ########################################
    meta = scattering.meta()
    S_all = {}
    for i, s in enumerate(Scx):
        if not np.isnan(meta['n'][i][1]):
            S_all[i] = s

    ref_shape = list(S_all.values())[0]['coef'].shape
    for i, s in S_all.items():
        assert s['coef'].shape == ref_shape, "{} != {} (n={})".format(
            s['coef'].shape, ref_shape, tuple(meta['n'][i]))


if __name__ == '__main__':
    if run_without_pytest:
        # test_alignment()
        test_shapes()
    else:
        pytest.main([__file__, "-s"])
