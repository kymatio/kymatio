from ...backend.numpy_backend import NumpyBackend


class NumpyBackend3D(NumpyBackend):
    @classmethod
    def modulus_rotation(cls, x, module=None):
        """Used for computing rotation invariant scattering transform coefficents.

            Parameters
            ----------
            x : tensor
                Size (batchsize, M, N, O, 2).
            module : tensor
                Tensor that holds the overall sum. If none, initializes the tensor
                to zero (default).
            Returns
            -------
            output : torch tensor
                Tensor of the same size as input_array. It holds the output of
                the operation::
                $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
                which is covariant to 3D translations and rotations.
        """
        if module is None:
            module = cls._np.abs(x) ** 2
        else:
            module = module ** 2 + cls._np.abs(x) ** 2
        return cls._np.sqrt(module)

    @classmethod
    def compute_integrals(cls, input_array, integral_powers):
        """Computes integrals.

            Computes integrals of the input_array to the given powers.
            Parameters
            ----------
            input_array : torch tensor
                Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
                dims.
            integral_powers : list
                List of P positive floats containing the p values used to
                compute the integrals of the input_array to the power p (l_p
                norms).
            Returns
            -------
            integrals : torch tensor
                Tensor of size (B, P) containing the integrals of the input_array
                to the powers p (l_p norms).
        """
        integrals = []
        for i_q, q in enumerate(integral_powers):
            integrals.append((input_array ** q).reshape((input_array.shape[0], -1)).sum(axis=1))
        integrals = cls._np.float32(cls._np.stack(integrals, axis=-1))
        return integrals

    @classmethod
    def stack(cls, arrays, L):
        S = cls._np.stack(arrays, axis=1)
        S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
        return S

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)
        return cls._fft.fftn(x, axes=(-3, -2, -1))

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)
        return cls._fft.ifftn(x, axes=(-3, -2, -1))

    @classmethod
    def cdgmm3d(cls, A, B):
        return cls.cdgmm(A, B)


backend = NumpyBackend3D
