from abc import ABCMeta, abstractmethod

class ScatteringNumpy(object, metaclass=ABCMeta):
   def __init__(self):
       super(ScatteringNumpy, self).__init__()

   @abstractmethod
   def build(self):
       raise NotImplementedError

   @abstractmethod
   def scattering(self, x):
       """ This function should call the functional scattering."""
       raise NotImplementedError

   def __call__(self, x):
       return self.scattering(x)