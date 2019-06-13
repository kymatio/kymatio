import scipy.fftpack
import warnings

def compute_padding(M, N, J):
    """
         Precomputes the future padded size. If 2^J=M or 2^J=N,
         the final size will be 2*M or 2*N. Border effects are unavoidable
         in this case, and it is likely that the input has either a
         compact support, either is periodic.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    if M > 2**J:
        M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    else:
        M_padded = 2*M

    if N > 2 ** J:
        N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J
    else:
        N_padded = 2*N

    return M_padded, N_padded

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)
