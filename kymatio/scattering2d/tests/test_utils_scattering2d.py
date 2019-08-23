import pytest

from kymatio import Scattering2D
from kymatio.scattering2d.frontend.numpy_frontend import Scattering2DNumpy

# Check that the default frontend is numpy and that errors are correctly launched.
def test_scattering2d_frontend():
    scattering = Scattering2D(2, shape=(10, 10))
    assert isinstance(scattering, Scattering2DNumpy), 'NumPy frontend is not selected by default'

    with pytest.raises(ImportError) as ve:
        scattering = Scattering2D(2, shape=(10, 10), frontend='doesnotexist')
    assert "module named" in ve.value.args[0]