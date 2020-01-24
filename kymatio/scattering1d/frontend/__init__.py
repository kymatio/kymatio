from ...frontend import Scattering

class Scattering1D(Scattering):
    def __init__(self, *args, **kwargs):
        super().__init__(name='1D', class_name='scattering1d', *args, **kwargs)


__all__ = ['Scattering1D']
