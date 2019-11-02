
class ScatteringNumPy(object):
    def __init__(self):
        super(ScatteringNumPy, self).__init__()
        
    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def __call__(self, x):
        """ This function should call the functional scattering."""
        return self.scattering(x)

