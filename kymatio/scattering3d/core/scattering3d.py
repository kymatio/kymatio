def scattering3d(x, filters, rotation_covariant, L, J, max_order, backend, averaging):
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
        stacked along the feature axis
    """
    rfft = backend.rfft
    ifft = backend.ifft
    cdgmm3d = backend.cdgmm3d
    modulus = backend.modulus
    modulus_rotation = backend.modulus_rotation
    stack = backend.stack

    U_0_c = rfft(x)

    s_order_1, s_order_2 = [], []
    for l in range(L + 1):
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J + 1):
            U_1_m = None
            if rotation_covariant:
                for m in range(len(filters[l][j_1])):
                    U_1_c = cdgmm3d(U_0_c, filters[l][j_1][m])
                    U_1_c = ifft(U_1_c)
                    U_1_m = modulus_rotation(U_1_c, U_1_m)
            else:
                U_1_c = cdgmm3d(U_0_c, filters[l][j_1][0])
                U_1_c = ifft(U_1_c)
                U_1_m = modulus(U_1_c)

            S_1_l = averaging(U_1_m)
            s_order_1_l.append(S_1_l)

            if max_order > 1:
                U_1_c = rfft(U_1_m)
                for j_2 in range(j_1 + 1, J + 1):
                    U_2_m = None
                    if rotation_covariant:
                        for m in range(len(filters[l][j_2])):
                            U_2_c = cdgmm3d(U_1_c, filters[l][j_2][m])
                            U_2_c = ifft(U_2_c)
                            U_2_m = modulus_rotation(U_2_c, U_2_m)
                    else:
                        U_2_c = cdgmm3d(U_1_c, filters[l][j_2][0])
                        U_2_c = ifft(U_2_c)
                        U_2_m = modulus(U_2_c)
                    S_2_l = averaging(U_2_m)
                    s_order_2_l.append(S_2_l)

        s_order_1.append(s_order_1_l)

        if max_order == 2:
            s_order_2.append(s_order_2_l)

    # Stack the orders (along the j axis) if needed.
    S = s_order_1
    if max_order == 2:
        S = [x + y for x, y in zip(S, s_order_2)]

    # Invert (ell, m × j) ordering to (m × j, ell).
    S = [x for y in zip(*S) for x in y]

    S = stack(S, L)

    return S
