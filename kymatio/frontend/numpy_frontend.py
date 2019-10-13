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
       
