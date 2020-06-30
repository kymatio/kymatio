from kymatio import Scattering3D, Scattering2D
import torch


def test_scattering3d():
    S = Scattering3D(3, (64, 64, 64)).cuda()
    x = torch.zeros((1, 64, 64, 64)).cuda().double()
    x[:, 16:48, 16:48, 16:48] = 1
    
    Sx = S(x)
