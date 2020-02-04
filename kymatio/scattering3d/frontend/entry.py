from ...frontend.entry import ScatteringEntry


class HarmonicScatteringEntry3D(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(name='harmonic 3D',
                         class_name='scattering3d',
                         *args, **kwargs)


__all__ = ['HarmonicScatteringEntry3D']
