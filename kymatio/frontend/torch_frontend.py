import torch.nn as nn

class ScatteringTorch(nn.Module):
    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""

        self.backend.input_checks(x)

        return self.scattering(x)

    _doc_array = 'torch.Tensor'
    _doc_array_n = ''

    _doc_alias_name = 'forward'

    _doc_alias_call = '.forward({x})'

    _doc_frontend_paragraph = r"""

        This class inherits from `torch.nn.Module`. As a result, it has all
        the same capabilities, including transferring the object to the GPU
        using the `cuda` or `to` methods. This object would then take GPU
        tensors as input and output the scattering coefficients of those
        tensors."""

    _doc_sample = 'torch.randn({shape})'

    _doc_has_shape = True

    _doc_has_out_type = True
