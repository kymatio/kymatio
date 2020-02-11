

class FFT:
    def __init__(self, fft, rfft, ifft, irfft, type_checks, real_check, complex_check):
        self.fft = fft
        self.rfft = rfft
        self.ifft = ifft
        self.irfft = irfft
        self.sanity_checks = type_checks
        self.real_check = real_check
        self.complex_check = complex_check

    def fft_forward(self, x, direction='C2C', inverse=False):
        """Interface with FFT routines for any dimensional signals and any backend signals.

            Example (for Torch)
            -------
            x = torch.randn(128, 32, 32, 2)
            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)

            Parameters
            ----------
            x : input
                Complex input for the FFT.
            direction : string
                'C2R' for complex to real, 'C2C' for complex to complex.
            inverse : bool
                True for computing the inverse FFT.
                NB : If direction is equal to 'C2R', then an error is raised.

            Raises
            ------
            RuntimeError
                In the event that we are going from complex to real and not doing
                the inverse FFT or in the event x is not contiguous.


            Returns
            -------
            output :
                Result of FFT or IFFT.
        """
        if direction == 'C2R':
            if not inverse:
                raise RuntimeError('C2R mode can only be done with an inverse FFT.')

        if direction == 'R2C':
            if inverse:
                raise RuntimeError('R2C mode can only be done with a FFT.')

        self.sanity_checks(x)

        if direction == 'C2R':
            self.complex_check(x)
            output = self.irfft(x)
        elif direction == 'C2C':
            self.complex_check(x)
            if inverse:
                output = self.ifft(x)
            else:
                output = self.fft(x)
        elif direction == 'R2C':
            self.real_check(x)
            output = self.rfft(x)

        return output

    def __call__(self, x, direction='C2C', inverse=False):
        return self.fft_forward(x, direction=direction, inverse=inverse)
