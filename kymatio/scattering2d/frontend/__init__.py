from ...frontend import Scattering

class Scattering2D(Scattering):
    def __init__(self, *args, **kwargs):
        super().__init__(name='2D', class_name='scattering2d', *args, **kwargs)


__all__ = ['Scattering2D']
