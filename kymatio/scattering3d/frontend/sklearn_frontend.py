from ...frontend.sklearn_frontend import ScatteringTransformerMixin
from ...numpy import HarmonicScattering3D as HarmonicScatteringNumPy3D


# NOTE: Order in base classes matters here, since we want the sklearn-specific
# documentation parameters to take precedence over NP.
class HarmonicScatteringTransformer3D(ScatteringTransformerMixin,
                                      HarmonicScatteringNumPy3D):
    pass


HarmonicScatteringTransformer3D._document()


__all__ = ['HarmonicScatteringTransformer3D']
