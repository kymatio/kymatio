# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg
import torch
def scattering3d(_input, filters, gaussian_filters, rotation_covariant, points, integral_powers, L, J, method,\
                 max_order, backend, averaging):
    """
    The forward pass of 3D solid harmonic scattering

    Parameters
    ----------
    input_array: torch tensor
        input of size (batchsize, M, N, O)

    Returns
    -------
    output: tuple | torch tensor
        if max_order is 1 it returns a torch tensor with the
        first order scattering coefficients
        if max_order is 2 it returns a torch tensor with the
        first and second order scattering coefficients,
        concatenated along the feature axis
    """
    finalize, fft, cdgmm3d, modulus, modulus_rotation = backend.finalize, backend.fft, backend.cdgmm3d, backend.modulus\
    , backend.modulus_rotation

    U_0_c = fft(_input)

    s_order_1, s_order_2 = [], []
    for l in range(L+1):
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J+1):
            U_1_m = None
            if rotation_covariant:
                for m in range(len(filters[l][j_1])):
                    U_1_c = cdgmm3d(U_0_c, filters[l][j_1][m])
                    U_1_c = fft(U_1_c, inverse=True)
                    U_1_m = modulus_rotation(U_1_c, U_1_m)
            else:
                U_1_c = cdgmm3d(U_0_c, filters[l][j_1][0])
                U_1_c = fft(U_1_c , inverse=True)
                U_1_m = modulus(U_1_c)

            U_1_c = fft(U_1_m)
            S_1_l = averaging(U_1_c, j_1)
            s_order_1_l.append(S_1_l)

            if max_order >1:
                for j_2 in range(j_1+1, J+1):
                    U_2_m = None
                    if rotation_covariant:
                        for m in range(len(filters[l][j_2])):
                            U_2_c = cdgmm3d(U_1_c, filters[l][j_2][m])
                            U_2_c = fft(U_2_c, inverse=True)
                            U_2_m = modulus_rotation(U_2_c, U_2_m)
                    else:
                        U_2_c = cdgmm3d(U_1_c, filters[l][j_2][0])
                        U_2_c = fft(U_2_c, inverse=True)
                        U_2_m = modulus(U_2_c)
                    U_2_c = fft(U_2_m)
                    S_2_l = averaging(U_2_c, j_2)
                    s_order_2_l.append(S_2_l)
        s_order_1.append(torch.stack([arr[..., 0] for arr in s_order_1_l], 1))
        if max_order == 2:
            s_order_2.append(torch.stack([arr[..., 0] for arr in s_order_2_l], 1))


    return finalize(s_order_1, s_order_2, max_order)



