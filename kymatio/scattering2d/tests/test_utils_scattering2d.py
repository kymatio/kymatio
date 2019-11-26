import pytest

from kymatio import Scattering2D
from kymatio.scattering2d.frontend.torch_frontend import Scattering2DTorch

# Check that the default frontend is Torch and that errors are correctly launched.
def test_scattering2d_frontend():
    scattering = Scattering2D(2, shape=(10, 10))
    assert isinstance(scattering, Scattering2DTorch), 'Torch frontend is not selected by default'

    with pytest.raises(ImportError) as ve:
        scattering = Scattering2D(2, shape=(10, 10), frontend='doesnotexist')
    assert "module named" in ve.value.args[0]

# Check the default backend is Torch and that errors are correctly launched.
def test_scattering2d_backend():
    with pytest.raises(ImportError) as ve:
        scattering = Scattering2D(2, shape=(10, 10), backend='doesnotexist')
    assert "can not be called" in ve.value.args[0]
