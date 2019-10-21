import torch
import torch.nn as nn
from packaging import version

assert version.parse(torch.__version__)>=version.parse("1.3"), 'Current PyTorch version is ' + str(torch.__version__) +\
                                                    ' . Please upgrade PyTorch to 1.3 at least.'

class ScatteringTorch(nn.Module):
    def __init__(self):
        super(ScatteringTorch, self).__init__()
   
    def build(self):
        """ Defines elementary routines. 
        This function should always call and create the filters via
        self.create_and_register_filters() defined below. For instance, via:
        self.filters = self.create_and_register_filters() """
        raise NotImplementedError
    
    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays as module's buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    def forward(self, x):
        return self.scattering(x)
    
    def loginfo(self):
        """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError

