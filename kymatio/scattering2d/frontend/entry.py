from ...frontend.entry import ScatteringEntry

class ScatteringEntry2D(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(name='2D', class_name='scattering2d', *args, **kwargs)


__all__ = ['ScatteringEntry2D']
