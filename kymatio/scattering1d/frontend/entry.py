from ...frontend.entry import ScatteringEntry

class ScatteringEntry1D(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(name='1D', class_name='scattering1d', *args, **kwargs)


__all__ = ['ScatteringEntry1D']
