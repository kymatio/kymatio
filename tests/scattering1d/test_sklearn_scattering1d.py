import numpy as np
import pytest
from sklearn.base import clone

from kymatio.sklearn import Scattering1D, TimeFrequencyScattering


@pytest.mark.parametrize('settings', [
    {}, {'T': 'global'}, {'oversampling': 1}, {'stride': 4},
])
def test_scattering1d_clone(settings):
    scattering = Scattering1D(J=4, shape=256, Q=2, **settings)
    copied = clone(scattering)
    x = np.linspace(-1, 1, 512).reshape(2, 256)

    assert copied.average == scattering.average
    np.testing.assert_array_equal(copied.transform(x), scattering.transform(x))


@pytest.mark.parametrize('settings', [
    {}, {'T': 'global'}, {'F': 'global'}, {'F': 0},
    {'T': 'global', 'F': 'global'},
])
def test_timefrequency_scattering_clone(settings):
    scattering = TimeFrequencyScattering(
        J=4, J_fr=2, Q=2, shape=256, **settings)
    copied = clone(scattering)
    x = np.linspace(-1, 1, 512).reshape(2, 256)

    assert copied.average == scattering.average
    assert copied.average_fr == scattering.average_fr
    np.testing.assert_array_equal(copied.transform(x), scattering.transform(x))
