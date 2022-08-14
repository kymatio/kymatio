import pytest
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase

def test_jtfs_build():
    # Test __init__
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=3, shape=4096, Q=8, backend='torch')
    assert jtfs.F is None

    # Test parse_T(self.F, self.J_fr, N_input_fr, T_alias='F')
    jtfs.build()
    assert jtfs.F == (2 ** jtfs.J_fr)
    assert jtfs.log2_F == jtfs.J_fr

    # Test that frequential padding is at least six times larger than F
    N_input_fr = len(jtfs.psi1_f)
    assert jtfs._N_padded_fr > (N_input_fr + 6 * jtfs.F)

    # Test that padded frequency domain is divisible by max subsampling factor
    assert (jtfs._N_padded_fr % (2**jtfs.J_fr)) == 0
