# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

def _fft_convolve(input_array, filter_array, fft, cdgmm3d):
    """
    Computes the fourier space convolution of the input_array,
    given in signal space, with a filter, given in fourier space.

    Parameters
    ----------

    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    filter_array: torch tensor
        size (M, N, O, 2)

    Returns
    -------

    output: the result of the convolution of input_array with filter

    """
    return fft(cdgmm3d(fft(input_array, inverse=False), filter_array), inverse=True)


def _low_pass_filter(input_array, low_pass):
    """
    Computes the convolution of input_array with a lowpass filter phi_j

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)

    j: int

    Returns
    -------
    output: the result of input_array :math:`\\star phi_J`

    """
    #low_pass = self.gaussian_filters[j]
    return _fft_convolve(input_array, low_pass)


def _compute_standard_scattering_coefs(input_array, low_pass, J, subsample):
    """
    Computes the convolution of input_array with a lowpass filter phi_J
    and downsamples by a factor J.

    Parameters
    ----------
    input_array: torch tensor of size (batchsize, M, N, O, 2)

    Returns
    -------
    output: the result of input_array \\star phi_J downsampled by a factor J

    """
    convolved_input = _low_pass_filter(input_array, low_pass)
    return subsample(convolved_input, J)


def _convolution_and_modulus(input_array, filters_l_m_j, complex_modulus):
    """
    Computes the convolution with a set of solid harmonics of scale j and
    degree l and returns the square root of their squared sum over m

    Parameters
    ----------
    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    l : int
        solid harmonic degree l
    j : int
        solid harmonic scale j
    m : int, optional
        solid harmonic rank m (defaults to 0)

    Returns
    -------
    output: torch tensor
            tensor of the same size as input_array. It holds the output of the
            operation::

            .. math:: \\text{input}_\\text{array} \\star \\psi_{j,l,m})

    """
    #filters_l_m_j = self.filters[l][j][m]
    return complex_modulus(_fft_convolve(input_array, filters_l_m_j))


def _compute_local_scattering_coefs(input_array, low_pass, points):
    """
    Computes the convolution of input_array with a lowpass filter phi_j and
    and returns the value of the output at particular points

    Parameters
    ----------
    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    points: torch tensor
        size (batchsize, number of points, 3)
    j: int
        the lowpass scale j of phi_j

    Returns
    -------
    output: torch tensor of size (batchsize, number of points, 1) with
            the values of the lowpass filtered moduli at the points given.

    """
    local_coefs = torch.zeros(input_array.size(0), points.size(1), 1)
    convolved_input = _low_pass_filter(input_array, low_pass)#j + 1)
    for i in range(input_array.size(0)):
        for j in range(points[i].size(0)):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs


def compute_scattering_coefs(input_array, method, args, filter, compute_integrals):
    """
    Computes the scattering coefficients out with any of the three methods
    'standard', 'local', 'integral'

    Parameters
    ----------
    input_array : torch tensor
        size (batchsize, M, N, O, 2)
    method : string
        method name with three possible values ("standard", "local", "integral")
    args : dict
        method specific arguments. It methods is equal to "standard", then one
        expects the array args['integral_powers'] to be a list that holds
        the exponents the moduli. It should be raised to before calculating
        the integrals. If method is equal to "local", args['points'] must contain
        a torch tensor of size (batchsize, number of points, 3) the points in
        coordinate space at which you want the moduli sampled
    j : int
        lowpass scale j of :math:`\\phi_j`

    Returns
    -------
    output: torch tensor
        The scattering coefficients as given by different methods.

    """
    methods = ['standard', 'local', 'integral']
    if (not method in methods):
        raise (ValueError('method must be in {}'.format(methods)))
    if method == 'integral':
        return compute_integrals(input_array[..., 0],
                                 args['integral_powers'])
    elif method == 'local':
        return _compute_local_scattering_coefs(input_array, args['points'], filter)
    elif method == 'standard':
        return _compute_standard_scattering_coefs(input_array)


def _rotation_covariant_convolution_and_modulus(input_array, filters_l_j):
    """
    Computes the convolution with a set of solid harmonics of scale j and
    degree l and returns the square root of their squared sum over m

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)
    l : int
        solid harmonic degree l

    j : int
        solid harmonic scale j

    Returns
    -------

    output : torch tensor
        tensor of the same size as input_array. It holds the output of
        the operation::

        $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

        which is covariant to 3D translations and rotations

    """
    #filters_l_j = self.filters[l][j]
    convolution_modulus = torch.zeros_like(input_array)
    for m in range(filters_l_j.size(0)):
        convolution_modulus[..., 0] += (_fft_convolve(
            input_array, filters_l_j[m]) ** 2).sum(-1)
    return torch.sqrt(convolution_modulus)


def scattering3d(_input, filters, gaussian_filters, rotation_covariant, points, integral_powers, L, J, method, max_order, backend):
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
    finalize = backend.finalize

    if rotation_covariant:
        convolution_and_modulus =_rotation_covariant_convolution_and_modulus
    else:
        convolution_and_modulus = _convolution_and_modulus

    s_order_1 = []
    s_order_2 = []

    method_args = dict(points=points,
                       integral_powers=integral_powers)

    for l in range(L+1):
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J+1):
            conv_modulus = convolution_and_modulus(_input, filters[l][j_1][0])
            s_order_1_l.append(compute_scattering_coefs(
                conv_modulus, method, method_args, j_1))
            if max_order == 1:
                continue
            for j_2 in range(j_1+1, J+1):
                conv_modulus_2 = convolution_and_modulus(
                    conv_modulus, filters[l][j_2][0])
                s_order_2_l.append(compute_scattering_coefs(
                    conv_modulus_2, method, method_args, j_2))
        s_order_1.append([arr[..., 0] for arr in s_order_1_l])
        if max_order == 2:
            s_order_2.append([arr[..., 0] for arr in s_order_2_l])

    finalize(s_order_1, s_order_2, max_order)
    #if max_order == 2:
    #    return backend.finalize(s_order_1, s_order_2)torch.cat(
    #        [torch.stack(s_order_1, dim=2),
    #        torch.stack(s_order_2, dim=2)], 1)
    #else:
    #    return torch.stack(s_order_1, dim=2)


