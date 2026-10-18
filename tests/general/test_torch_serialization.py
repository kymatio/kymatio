import pickle

import pytest
import torch

from kymatio.torch import (
    HarmonicScattering3D, Scattering1D, Scattering2D, TimeFrequencyScattering,
)


@pytest.mark.parametrize('protocol', [2, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('scattering_class,kwargs', [
    (Scattering1D, dict(J=2, shape=64, Q=1)),
    (Scattering2D, dict(J=2, shape=(32, 32), L=4)),
    (TimeFrequencyScattering, dict(J=2, J_fr=1, shape=64, Q=1)),
    (HarmonicScattering3D, dict(J=1, shape=(8, 8, 8), L=1)),
])
def test_torch_scattering_pickle(scattering_class, kwargs, dtype, protocol):
    scattering = scattering_class(**kwargs).to(dtype=dtype).eval()

    restored = pickle.loads(pickle.dumps(scattering, protocol=protocol))

    assert type(restored) is type(scattering)
    assert restored.training == scattering.training
    original_buffers = dict(scattering.named_buffers())
    restored_buffers = dict(restored.named_buffers())
    assert original_buffers.keys() == restored_buffers.keys()
    for name, original in original_buffers.items():
        assert torch.equal(original, restored_buffers[name])
        assert original.dtype == restored_buffers[name].dtype

    shape = kwargs['shape']
    if isinstance(shape, int):
        shape = (shape,)
    x = torch.randn((2,) + shape, dtype=dtype)
    assert torch.equal(restored(x), scattering(x))
