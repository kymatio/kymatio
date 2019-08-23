from abc import ABCMeta, abstractmethod
import torch.nn as nn

class ScatteringTorch(nn.Module, metaclass=ABCMeta):
   def __init__(self):
       super(ScatteringTorch, self).__init__()

   @abstractmethod
   def build(self):
       """ Defines elementary routines. This function should always call and create the filters via
       self.create_and_register_filters() defined below. For instance, via:
       self.filters = self.create_and_register_filters() """

   @abstractmethod
   def create_and_register_filters(self):
       """ This function should run a filterbank function that
       will create the filters as numpy array, and then, it should
       save those arrays as module's buffers."""

   @abstractmethod
   def scattering(self, x):
       """ This function should call the functional scattering."""

   def forward(self, x):
      return self.scattering(x)