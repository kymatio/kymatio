from ...frontend.sklearn_frontend import ScatteringTransformerMixin
from ...numpy import Scattering2D as ScatteringNumPy2D


# NOTE: Order in base classes matters here, since we want the sklearn-specific
# documentation parameters to take precedence over NP.
class ScatteringTransformer2D(ScatteringTransformerMixin, ScatteringNumPy2D):
    pass


ScatteringTransformer2D._document()


__all__ = ['ScatteringTransformer2D']
