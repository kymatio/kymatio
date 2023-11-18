import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(
        self,
        J,
        shape,
        Q=1,
        T=None,
        stride=None,
        max_order=2,
        oversampling=0,
        out_type="array",
        backend="torch",
    ):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(
            self, J, shape, Q, T, stride, max_order, oversampling, out_type, backend
        )
        ScatteringBase1D._instantiate_backend(self, "kymatio.scattering1d.backend.")
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.register_filters()

    def register_filters(self):
        """This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        for level in range(len(self.phi_f["levels"])):
            self.phi_f["levels"][level] = (
                torch.from_numpy(self.phi_f["levels"][level]).float().view(-1, 1)
            )
            self.register_buffer("tensor" + str(n), self.phi_f["levels"][level])
            n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = (
                    torch.from_numpy(psi_f["levels"][level]).float().view(-1, 1)
                )
                self.register_buffer("tensor" + str(n), psi_f["levels"][level])
                n += 1
        for psi_f in self.psi2_f:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = (
                    torch.from_numpy(psi_f["levels"][level]).float().view(-1, 1)
                )
                self.register_buffer("tensor" + str(n), psi_f["levels"][level])
                n += 1
        return n

    def load_filters(self):
        """This function loads filters from the module's buffer"""
        buffer_dict = dict(self.named_buffers())
        n = 0

        for level in range(len(self.phi_f["levels"])):
            self.phi_f["levels"][level] = buffer_dict["tensor" + str(n)]
            n += 1

        for psi_f in self.psi1_f:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = buffer_dict["tensor" + str(n)]
                n += 1

        for psi_f in self.psi2_f:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = buffer_dict["tensor" + str(n)]
                n += 1
        return n

    def scattering(self, x):
        self.load_filters()
        return super().scattering(x)


ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch(ScatteringTorch1D, TimeFrequencyScatteringBase):
    def __init__(
        self,
        J,
        J_fr,
        Q,
        shape,
        T=None,
        stride=None,
        Q_fr=1,
        F=None,
        stride_fr=None,
        out_type="array",
        format="time",
        backend="torch"
    ):
        ScatteringTorch.__init__(self)
        TimeFrequencyScatteringBase.__init__(
            self,
            J=J,
            J_fr=J_fr,
            Q=Q,
            shape=shape,
            T=T,
            stride=stride,
            Q_fr=Q_fr,
            F=F,
            stride_fr=stride_fr,
            out_type=out_type,
            format=format,
            backend=backend,
        )
        ScatteringBase1D._instantiate_backend(self, "kymatio.scattering1d.backend.")
        TimeFrequencyScatteringBase.build(self)
        TimeFrequencyScatteringBase.create_filters(self)
        self.register_filters()

    def register_filters(self):
        n = super(TimeFrequencyScatteringTorch, self).register_filters()
        for level in range(len(self.filters_fr[0]["levels"])):
            self.filters_fr[0]["levels"][level] = (
                torch.from_numpy(self.filters_fr[0]["levels"][level])
                .float()
                .view(-1, 1)
            )
            self.register_buffer("tensor" + str(n), self.filters_fr[0]["levels"][level])
            n += 1
        for psi_f in self.filters_fr[1]:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = (
                    torch.from_numpy(psi_f["levels"][level]).float().view(-1, 1)
                )
                self.register_buffer("tensor" + str(n), psi_f["levels"][level])
                n += 1

    def load_filters(self):
        buffer_dict = dict(self.named_buffers())
        n = super(TimeFrequencyScatteringTorch, self).load_filters()

        for level in range(len(self.filters_fr[0]["levels"])):
            self.filters_fr[0]["levels"][level] = buffer_dict["tensor" + str(n)]
            n += 1

        for psi_f in self.filters_fr[1]:
            for level in range(len(psi_f["levels"])):
                psi_f["levels"][level] = buffer_dict["tensor" + str(n)]
                n += 1


TimeFrequencyScatteringTorch._document()


__all__ = ["ScatteringTorch1D", "TimeFrequencyScatteringTorch"]
