from sklearn.base import BaseEstimator, TransformerMixin

from kymatio.numpy import Scattering1D as ScatteringNumPy1D
from kymatio.numpy import Scattering2D as ScatteringNumPy2D
from kymatio.numpy import HarmonicScattering3D as HarmonicScatteringNumPy3D


class ScatteringTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, x=None, y=None):
        # No fitting necessary.
        return self

    def predict(self, x):
        n_samples = x.shape[0]

        x = x.reshape((-1,) + self.shape)

        Sx = self.scattering(x)

        Sx = Sx.reshape(n_samples, -1)

        return Sx

    transform = predict


class ScatteringTransformer1D(ScatteringNumPy1D, ScatteringTransformerMixin):
    pass


class ScatteringTransformer2D(ScatteringNumPy2D, ScatteringTransformerMixin):
    pass


class HarmonicScatteringTransformer3D(HarmonicScatteringNumPy3D,
                                      ScatteringTransformerMixin):
    pass


Scattering1D = ScatteringTransformer1D
Scattering2D = ScatteringTransformer2D
HarmonicScattering3D = HarmonicScatteringTransformer3D


__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
