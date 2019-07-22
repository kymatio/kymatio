__all__ = ['torch_backend', 'torch_skcuda_backend', 'numpy_backend']

try:
    from .torch_backend import backend as torch_backend
    from .torch_skcuda_backend import backend as torch_skcuda_backend
    from .numpy_backend import backend as numpy_backend
except:
    pass