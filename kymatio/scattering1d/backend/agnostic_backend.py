
def pad(x, pad_left, pad_right, pad_mode='reflect', axis=-1,
        backend_name='numpy'):
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
    backend_name : str
        One of: numpy, torch, tensorflow

    Returns
    -------
    out : type(x)
        Padded tensor.
    """
    if backend_name == 'numpy':
        import numpy as np
        backend = np
        kw = dict(dtype=x.dtype)
    elif backend_name == 'torch':
        import torch
        backend = torch
        kw = dict(dtype=x.dtype, layout=x.layout, device=x.device)
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        backend = tf
        kw = dict(dtype=x.dtype)

    N = x.shape[axis]
    axis_idx = axis if axis >= 0 else (x.ndim + axis)
    padded_shape = (x.shape[:axis_idx] + (N + pad_left + pad_right,) +
                    x.shape[axis_idx + 1:])
    out = backend.zeros(padded_shape, **kw)
    if backend_name == 'tensorflow':  # TODO
        out = out.numpy()

    pad_right_idx = (out.shape[axis] -
                     (pad_right if pad_right != 0 else None))
    out[index_axis(pad_left, pad_right_idx, axis, x.ndim)] = x

    if pad_mode == 'zero':
        pass  # already done
    elif pad_mode == 'reflect':
        xflip = _flip(x, backend, axis, backend_name)
        out = _pad_reflect(x, xflip, out, pad_left, pad_right, N, axis)

    if backend_name == 'tensorflow':  # TODO
        out = tf.constant(out)
    return out


def _flip(x, backend, axis, backend_name='numpy'):
    if backend_name in ('numpy', 'tensorflow'):
        return x[flip_axis(axis, x.ndim)]
    elif backend_name == 'torch':
        axis = axis if axis >= 0 else (x.ndim + axis)
        return backend.flip(x, (axis,))


def _pad_reflect(x, xflip, out, pad_left, pad_right, N, axis):
    # fill maximum needed number of reflections
    # pad left ###############################################################
    def idx(i0, i1):
        return index_axis(i0, i1, axis, x.ndim)

    step, already_stepped  = 0, 0
    while already_stepped < pad_left:
        max_step = min(N - 1, pad_left - already_stepped)
        end = pad_left - already_stepped
        start = end - max_step
        if step % 2 == 0:
            out[idx(start, end)] = xflip[idx(-(max_step + 1), -1)]
        else:
            out[idx(start, end)] = x[idx(-(max_step + 1), -1)]
        step += 1
        already_stepped += max_step

    # pad right ##############################################################
    step, already_stepped  = 0, 0
    while already_stepped < pad_right:
        max_step = min(N - 1, pad_right - already_stepped)
        start = N + pad_left + already_stepped
        end = start + max_step
        if step % 2 == 0:
            out[idx(start, end)] = xflip[idx(1, max_step + 1)]
        else:
            out[idx(start, end)] = x[idx(1, max_step + 1)]
        step += 1
        already_stepped += max_step
    return out


def index_axis(i0, i1, axis, ndim, step=1):
    if axis < 0:
        slc = (slice(None),) * (ndim + axis) + (slice(i0, i1, step),)
    else:
        slc = (slice(None),) * axis + (slice(i0, i1, step),)
    return slc


def flip_axis(axis, ndim, step=1):
    return index_axis(-1, None, axis, ndim, -step)
