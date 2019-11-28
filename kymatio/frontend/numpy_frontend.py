class ScatteringNumPy(object):
    def __init__(self):
        self.frontend_name = 'numpy'

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def __call__(self, x):
        """ This function provides a standard NumPy calling interface to the
        scattering computation."""
        return self.scattering(x)
