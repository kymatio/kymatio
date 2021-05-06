
def pad(x, pad_left, pad_right, pad_mode='reflect', backend_name='numpy'):
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

    N = x.shape[-1]
    out = backend.zeros(x.shape[:-1] + (N + pad_left + pad_right,), **kw)
    if backend_name == 'tensorflow':  # TODO
        out = out.numpy()
    out[..., pad_left:-pad_right] = x

    if pad_mode == 'zero':
        out[..., :pad_left] = 0
        out[..., -pad_right:] = 0
    elif pad_mode == 'reflect':
        xflip = _flip(x, backend, backend_name)
        out = _pad_reflect(x, xflip, out, pad_left, pad_right, N)

    if backend_name == 'tensorflow':  # TODO
        out = tf.constant(out)
    return out


def _flip(x, backend, backend_name='numpy'):
    if backend_name == 'numpy':
        return x[..., ::-1]
    elif backend_name == 'torch':
        return backend.flip(x, (x.ndim - 1,))
    elif backend_name == 'tensorflow':
        return x[..., ::-1]


def _pad_reflect(x, xflip, out, pad_left, pad_right, N):
    # fill maximum needed number of reflections
    # pad left ###############################################################
    step, already_stepped  = 0, 0
    while already_stepped < pad_left:
        max_step = min(N - 1, pad_left - already_stepped)
        end = pad_left - already_stepped
        start = end - max_step
        if step % 2 == 0:
            out[..., start:end] = xflip[..., -(max_step + 1):-1]
        else:
            out[..., start:end] = x[..., -(max_step + 1):-1]
        step += 1
        already_stepped += max_step

    # pad right ##############################################################
    step, already_stepped  = 0, 0
    while already_stepped < pad_right:
        max_step = min(N - 1, pad_right - already_stepped)
        start = N + pad_left + already_stepped
        end = start + max_step
        if step % 2 == 0:
            out[..., start:end] = xflip[..., 1:(max_step + 1)]
        else:
            out[..., start:end] = x[..., 1:(max_step + 1)]
        step += 1
        already_stepped += max_step
    return out
