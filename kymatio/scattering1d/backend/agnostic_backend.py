import math
from ...toolkit import _infer_backend


def pad(x, pad_left, pad_right, pad_mode='reflect', axis=-1, out=None):
    """Backend-agnostic padding.

    Parameters
    ----------
    x : input tensor
        Input tensor.
    pad_left : int >= 0
        Amount to pad on left.
    pad_right : int >= 0
        Amount to pad on right.
    pad_mode : str
        One of supported padding modes:
            - zero:    [0,0,0,0, 1,2,3,4, 0,0,0]
            - reflect: [3,4,3,2, 1,2,3,4, 3,2,1]
    axis : int
        Axis to pad.

    Returns
    -------
    out : type(x)
        Padded tensor.
    """
    import numpy as np

    if pad_mode == 'zero':
        pad_mode = 'constant'
    pad_widths = [[0, 0]] * x.ndim
    pad_widths[axis] = [pad_left, pad_right]

    # handle backend
    backend, backend_name = _infer_backend(x, get_name=True)
    if backend_name == 'torch':
        info = {k: getattr(x, k) for k in ('dtype', 'device')}
        x = x.cpu().detach().numpy()
        x = np.pad(x, pad_widths, mode=pad_mode)
        x = backend.tensor(x, **info)
    elif backend_name == 'tensorflow':
        info = {k: getattr(x, k) for k in ('dtype',)}
        x = np.pad(x, pad_widths, mode=pad_mode)
        x = backend.convert_to_tensor(x, **info)
    else:
        x = np.pad(x, pad_widths, mode=pad_mode)
    return x



def conj_reflections(backend, x, ind_start, ind_end, k, N, pad_left, pad_right,
                     trim_tm):
    """Conjugate reflections to mirror spins rather than negate them.

    Won't conjugate *at* boundaries:

        [..., 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        [..., 1, 2, 3, 4, 5, 6, 7,-6,-5,-4,-3,-2, 1, 2, 3, 4, 5, 6, 7]
    """
    return x


def unpad_dyadic(x, N, orig_padded_len, target_padded_len, k=0):
    current_log2_N = math.ceil(math.log2(N))
    current_N_dyadic = 2**current_log2_N

    J_pad = math.log2(orig_padded_len)
    current_padded = 2**(J_pad - k)
    current_to_pad_dyadic = current_padded - current_N_dyadic

    start_dyadic = current_to_pad_dyadic // 2
    end_dyadic = start_dyadic + current_N_dyadic

    log2_target_to_pad = target_padded_len - 2**current_log2_N
    # x[3072:3072+2048] -> x[3072-1024:3072+2048+1024]
    # == x[2048:6144]; len: 8192 --> 4096
    unpad_start = int(start_dyadic - log2_target_to_pad // 2)
    unpad_end = int(end_dyadic + log2_target_to_pad // 2)
    x = x[..., unpad_start:unpad_end]
    return x

## helpers ###################################################################
def index_axis(i0, i1, axis, ndim, step=1):
    if axis < 0:
        slc = (slice(None),) * (ndim + axis) + (slice(i0, i1, step),)
    else:
        slc = (slice(None),) * axis + (slice(i0, i1, step),)
    return slc


def index_axis_with_array(idxs, axis, ndim):
    if axis < 0:
        slc = (slice(None),) * (ndim + axis) + (idxs,)
    else:
        slc = (slice(None),) * axis + (idxs,)
    return slc


def flip_axis(axis, ndim, step=1):
    return index_axis(-1, None, axis, ndim, -step)


def stride_axis(step, axis, ndim):
    return index_axis(None, None, axis, ndim, step)
