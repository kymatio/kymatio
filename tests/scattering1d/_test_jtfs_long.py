"""Tests too long for CI so done locally."""
import pytest
import numpy as np
from kymatio import TimeFrequencyScattering1D
from timeit import default_timer as dtime

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 1
# whether to report testing times
verbose = 1


def test_average_combos():
    """Test that various combinations of `average` & `average_fr` don't error."""
    N = 2048
    x = np.random.randn(N)

    J, Q, J_fr, Q_fr = 10, 16, 4, 1

    if verbose:
        print("\n// test_average_combos:")
    total_time = 0
    for average in (True, False):
      for average_fr in (True, False):
        for oversampling in (0, 1, 2, 999):
          for oversampling_fr in (0, 1, 2, 999):
            t0 = dtime()
            params_str = ("(average, average_fr, oversampling, oversampling_fr) "
                          "= ({}, {}, {}, {})").format(
                              average, average_fr, oversampling, oversampling_fr)
            out_type = "array" if average and average_fr else "list"

            jtfs = TimeFrequencyScattering1D(
                shape=N, J=J, Q=Q, J_fr=J_fr, Q_fr=Q_fr,
                oversampling=oversampling, oversampling_fr=oversampling_fr,
                out_type=out_type, frontend=default_backend)
            try:
                _ = jtfs(x)
            except Exception as e:
                raise Exception("Failed on %s with \n%s" % (params_str, e))

            if not verbose:
                continue
            elapsed = (dtime() - t0)
            total_time += elapsed
            print("{:.3f}s elapsed ({:.3f}s total) for {}".format(
                elapsed, total_time, params_str))


if __name__ == '__main__':
    if run_without_pytest:
        test_average_combos()
        # test_long_input()  # TODO anything interesting here?
    else:
        pytest.main([__file__, "-s"])
