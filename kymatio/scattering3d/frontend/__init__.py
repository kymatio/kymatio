from ...frontend import Scattering


class HarmonicScattering3D(Scattering):
    def __init__(self, *args, **kwargs):
        super().__init__(name='harmonic 3D',
                         class_name='scattering3d',
                         *args, **kwargs)


__all__ = ['HarmonicScattering3D']
