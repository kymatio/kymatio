from ...frontend.sklearn_frontend import ScatteringTransformerMixin
from ...numpy import Scattering1D as ScatteringNumPy1D
from ...numpy import TimeFrequencyScattering as TimeFrequencyScatteringNumPy

# NOTE: Order in base classes matters here, since we want the sklearn-specific
# documentation parameters to take precedence over NP.
class ScatteringTransformer1D(ScatteringTransformerMixin, ScatteringNumPy1D):
    pass

ScatteringTransformer1D._document()

class TimeFrequencyScatteringTransformer(ScatteringTransformerMixin, TimeFrequencyScatteringNumPy):
    pass

TimeFrequencyScatteringTransformer._document()

__all__ = ['ScatteringTransformer1D', 'TimeFrequencyScatteringTransformer']