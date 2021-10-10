
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
    # handle backend
    backend, backend_name = _infer_backend(x, get_name=True)
    kw = {'dtype': x.dtype}
    if backend_name == 'torch':
        kw.update({'layout': x.layout, 'device': x.device})

    # extract shape info
    N = x.shape[axis]
    axis_idx = axis if axis >= 0 else (x.ndim + axis)
    padded_shape = (x.shape[:axis_idx] + (N + pad_left + pad_right,) +
                    x.shape[axis_idx + 1:])
    # make `out` if not provided / run assert
    if out is None:
        out = backend.zeros(padded_shape, **kw)
    else:
        assert out.shape == padded_shape, (out.shape, padded_shape)

    # fill initial `x`
    pad_right_idx = ((out.shape[axis] - pad_right) if pad_right != 0 else
                     None)
    ix = index_axis(pad_left, pad_right_idx, axis, x.ndim)
    if backend_name != 'tensorflow':
        out[ix] = x
    else:
        from kymatio.backend.tensorflow_backend import TensorFlowBackend as B
        out = B.assign_slice(out, x, ix)

    # do padding
    if pad_mode == 'zero':
        pass  # already done
    elif pad_mode == 'reflect':
        xflip = _flip(x, backend, axis, backend_name)
        out = _pad_reflect(x, xflip, out, pad_left, pad_right, N, axis,
                           backend_name)
    return out


def _flip(x, backend, axis, backend_name='numpy'):
    if backend_name in ('numpy', 'tensorflow'):
        return x[flip_axis(axis, x.ndim)]
    elif backend_name == 'torch':
        axis = axis if axis >= 0 else (x.ndim + axis)
        return backend.flip(x, (axis,))


def _pad_reflect(x, xflip, out, pad_left, pad_right, N, axis, backend_name):
    # fill maximum needed number of reflections
    # pad left ###############################################################
    def idx(i0, i1):
        return index_axis(i0, i1, axis, x.ndim)

    def assign(out, ix0, ix1):
        # handle separately for performance
        if backend_name != 'tensorflow':
            if step % 2 == 0:
                out[ix0] = xflip[ix1]
            else:
                out[ix0] = x[ix1]
        else:
            if step % 2 == 0:
                out = B.assign_slice(out, xflip[ix1], ix0)
            else:
                out = B.assign_slice(out, x[ix1], ix0)
        return out

    if backend_name == 'tensorflow':
        from kymatio.backend.tensorflow_backend import TensorFlowBackend as B

    step, already_stepped  = 0, 0
    while already_stepped < pad_left:
        max_step = min(N - 1, pad_left - already_stepped)
        end = pad_left - already_stepped
        start = end - max_step
        ix0, ix1 = idx(start, end), idx(-(max_step + 1), -1)
        out = assign(out, ix0, ix1)
        step += 1
        already_stepped += max_step

    # pad right ##############################################################
    step, already_stepped  = 0, 0
    while already_stepped < pad_right:
        max_step = min(N - 1, pad_right - already_stepped)
        start = N + pad_left + already_stepped
        end = start + max_step
        ix0, ix1 = idx(start, end), idx(1, max_step + 1)
        out = assign(out, ix0, ix1)
        step += 1
        already_stepped += max_step
    return out

## helpers ###################################################################
def index_axis(i0, i1, axis, ndim, step=1):
    if axis < 0:
        slc = (slice(None),) * (ndim + axis) + (slice(i0, i1, step),)
    else:
        slc = (slice(None),) * axis + (slice(i0, i1, step),)
    return slc


def flip_axis(axis, ndim, step=1):
    return index_axis(-1, None, axis, ndim, -step)


def _infer_backend(x, get_name=False):
    # this belongs in `kymatio\toolkit.py`
    import numpy as np

    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]

    module = type(x).__module__.split('.')[0]

    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    elif isinstance(x, (int, float)):
        # list of lists, fallback to numpy
        module = 'numpy'
        backend = np
    else:
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))
