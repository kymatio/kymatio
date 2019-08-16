from abc import ABCMeta, abstractmethod
import torch.nn as nn

class ScatteringTorch(nn.Module, metaclass=ABCMeta):
   def __init__(self):
       super(ScatteringTorch, self).__init__()

   @abstractmethod
   def build(self):
       self.filters = self.create_and_register_filters()
       raise NotImplementedError

   @abstractmethod
   def create_and_register_filters(self):
       """ This function should run a filterbank function that
       will create the filters as numpy array, and then, it should
       save those arrays as module's buffers."""
       raise NotImplementedError

   @abstractmethod
   def scattering(self, x):
       """ This function should call the functional scattering."""
       raise NotImplementedError

   def forward(self, x):
      return self.scattering(x)