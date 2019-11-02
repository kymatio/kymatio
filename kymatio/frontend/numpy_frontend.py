<<<<<<< HEAD

class ScatteringNumPy(object):
   def __init__(self):
       super(ScatteringNumPy, self).__init__()

   def scattering(self, x):
       """ This function should compute the scattering transform."""
        raise NotImplementedError

   def __call__(self, x):
       """ This function should call the functional scattering."""
       return self.scattering(x)

   def loginfo(self):
       """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError

=======
from abc import ABCMeta, abstractmethod

class ScatteringNumpy(object, metaclass=ABCMeta):
   def __init__(self):
       super(ScatteringNumpy, self).__init__()

   @abstractmethod
   def build(self):
       """ Defines elementary routines."""

   @abstractmethod
   def scattering(self, x):
       """ This function should call the functional scattering."""

   def __call__(self, x):
       return self.scattering(x)

   @abstractmethod
   def loginfo(self):
       """
       Returns the logging message when the frontend is deployed.
       -------

       """
       
>>>>>>> e801b80... adding all basic files
