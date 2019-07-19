import torch.nn as nn

class Scattering(nn.Module):
   def __init__(self, J, shape, max_order = None):
       super(Scattering, self).__init__()
       self.J = J
       self.shape = shape
       self.max_order = max_order
       self.build()
       raise NotImplementedError

   def build(self):
       self.filters = self.create_and_register_filters()
       raise NotImplementedError

   def create_and_register_filters(self):
       """ This function should run a filterbank function that
       will create the filters as numpy array, and then, it should
       save those arrays as module's buffers."""
       raise NotImplementedError

   def scattering(self, x):
       """ This function should call the functional scattering."""
       raise NotImplementedError

   def forward(self, x):
      return self.scattering(x)