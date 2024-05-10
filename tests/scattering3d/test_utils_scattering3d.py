from kymatio.scattering3d.utils import sqrt
import pytest
import warnings
import numpy as np

def test_utils():
    # Simple test
    x = np.arange(8)
    y = sqrt(x**2)
    assert (y == x).all()

    # Test problematic case
    x = np.arange(8193, dtype='float32')
    y = sqrt(x**2)
    assert (y == x).all()

    # Make sure we still don't let in negatives...
    with pytest.warns(RuntimeWarning) as record:
        x = np.array([-1, 0, 1])
        y = sqrt(x)
    assert "Negative" in record[0].message.args[0]

    # ...unless they are complex numbers!
    with warnings.catch_warnings(record=True) as record:
        x = np.array([-1, 0, 1], dtype='complex64')
        y = sqrt(x)
    assert len(record) == 0
