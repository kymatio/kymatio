import pytest
import numpy as np
from kymatio import Scattering1D


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

    N = 1024
    J = int(np.log2(N))
    common_params = dict(shape=N, J=J, frontend='numpy')
    th_above = 1e-2
    th_below = .5

    for Q in (1, 8, 16):
        test_params = dict(Q=Q, T=2**(common_params['J'] - 1))
        test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                    test_params.items())
        try:
            jtfs = Scattering1D(**common_params, **test_params)
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
