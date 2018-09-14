import numpy as np
import torch
from torch.autograd import Variable
from scattering import Scattering2D
from scattering.scattering2d.differentiable import scattering
from scattering.scattering2d.differentiable import prepare_padding_size, cast
from scattering.scattering2d.filters_bank import filters_bank


def test_output_similarity_differentiable():
    M, N = 32, 32
    J = 2

    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J)

    Psi = filters['psi']
    Phi = [filters['phi'][j] for j in range(J)]

    Psi, Phi = cast(Psi, Phi, torch.cuda.FloatTensor)

    input = Variable(torch.randn(1, 3, 32, 32).cuda())
    input.requires_grad = True

    S = scattering(input, Psi, Phi, J)
    S.sum().backward()

    scat = Scattering2D(M, N, J).cuda()
    S_scat = scat(input.data)
    
    assert np.allclose(scat(input.data).cpu().numpy(), S.data.cpu().numpy())


if __name__ == "__main__":
    test_output_similarity_differentiable()

