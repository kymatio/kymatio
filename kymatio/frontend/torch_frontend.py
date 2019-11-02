import torch
import torch.nn as nn
from packaging import version

assert version.parse(torch.__version__)>=version.parse("1.3"), 'Current PyTorch version is ' + str(torch.__version__) +\
                                                    ' . Please upgrade PyTorch to 1.3 at least.'

class ScatteringTorch(nn.Module):
    def __init__(self):
        super(ScatteringTorch, self).__init__()

    def register_backend(self, string):
        """ This function should register the backend to be used"""
        if not self.backend:
            backend = __import__(string, globals(), locals(), ['backend'], 0)
            self.backend = backend.backend
        elif not self.backend.name.startswith('torch'):
            raise RuntimeError('This backend is not supported.')
    
    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def forward(self, x):
        """ This function should call the functional scattering."""
        return self.scattering(x)
    
