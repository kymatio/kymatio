
class ScatteringNumPy(object):
   def __init__(self):
       super(ScatteringNumpy, self).__init__()

   def build(self):
       """ Defines elementary routines."""
        raise NotImplementedError

   def scattering(self, x):
       """ This function should compute the scattering transform."""
        raise NotImplementedError

   def __call__(self, x):
       """ This function should call the functional scattering."""
       return self.scattering(x)

   def loginfo(self):
       """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError

