from ...frontend.jax_frontend import ScatteringJax
from .numpy_frontend import ScatteringNumPy1D, TimeFrequencyScatteringNumPy
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


class ScatteringJax1D(ScatteringJax, ScatteringNumPy1D):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in ScatteringNumPy1D
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through ScatteringBase1D._instantiate_backend the jax backend will
    # be loaded

    def __init__(
        self,
        J,
        shape,
        Q=1,
        T=None,
        stride=None,
        max_order=2,
        oversampling=0,
        out_type="array",
        backend="jax",
    ):

        ScatteringJax.__init__(self)
        ScatteringBase1D.__init__(
            self, J, shape, Q, T, stride, max_order, oversampling, out_type, backend
        )
        ScatteringBase1D._instantiate_backend(self, "kymatio.scattering1d.backend.")
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)


ScatteringJax1D._document()


class TimeFrequencyScatteringJax(ScatteringJax, TimeFrequencyScatteringNumPy):
    # This class inherits the attribute "frontend" from ScatteringJax
    # It overrides the __init__ function present in TimeFrequencyScatteringNumPy
    # in order to add the default argument for backend and call the
    # ScatteringJax.__init__
    # Through TimeFrequencyScatteringBase._instantiate_backend the jax backend will
    # be loaded

    def __init__(
        self,
        J,
        J_fr,
        Q,
        shape,
        T=None,
        stride=None,
        Q_fr=1,
        F=None,
        stride_fr=None,
        out_type="array",
        format="time",
        backend="jax"
    ):

        ScatteringJax.__init__(self)
        TimeFrequencyScatteringBase.__init__(
            self,
            J=J,
            J_fr=J_fr,
            Q=Q,
            shape=shape,
            T=T,
            stride=stride,
            Q_fr=Q_fr,
            F=F,
            stride_fr=stride_fr,
            out_type=out_type,
            format=format,
            backend=backend,
        )
        ScatteringBase1D._instantiate_backend(self, "kymatio.scattering1d.backend.")
        TimeFrequencyScatteringBase.build(self)
        TimeFrequencyScatteringBase.create_filters(self)


TimeFrequencyScatteringJax._document()

__all__ = ["ScatteringJax1D", "TimeFrequencyScatteringJax"]
