from abc import ABCMeta, abstractmethod

class ScatteringNumPy(object, metaclass=ABCMeta):
   def __init__(self):
       super(ScatteringNumPy, self).__init__()

   @abstractmethod
   def build(self):
       """ Defines elementary routines."""

   @abstractmethod
   def scattering(self, x):
       """ This function should compute the scattering transform."""

   def __call__(self, x):
       """ This function should call the functional scattering."""
       return self.scattering(x)

   @abstractmethod
   def loginfo(self):
       """ Returns the logging message when the frontend is deployed."""

