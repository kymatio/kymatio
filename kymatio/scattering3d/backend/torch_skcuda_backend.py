import torch
import warnings
from skcuda import cublas
from string import Template

from ...backend.torch_skcuda_backend import TorchSkcudaBackend

from .torch_backend import TorchBackend3D


class TorchSkcudaBackend3D(TorchSkcudaBackend, TorchBackend3D):
    @classmethod
    def cdgmm3d(cls, A, B):
        return cls.cdgmm(A, B)


backend = TorchSkcudaBackend3D
