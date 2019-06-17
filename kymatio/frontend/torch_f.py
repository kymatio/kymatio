import torch.nn as nn

class Scattering(nn.Module):
   def __init__(self, J, shape, max_order = None):
       super(Scattering, self).__init__()
       return
       self.J = J
       self.shape = shape
       self.max_order = max_order
       self.build()

   def build(self):
       return
       self.filters = self.create_and_register_filters()

   def create_and_register_filters(self):
       pass

   def scattering(self):
       pass

   def forward(self, x):
      return self.scattering(x)