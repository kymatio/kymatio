from ...toolkit import _infer_backend
import numpy as np


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
    if out is not None:
        return pad_fast(x, pad_left, pad_right, out)

    backend, backend_name = _infer_backend(x, get_name=True)
    kw = {'dtype': x.dtype}
    if backend_name == 'torch':
        kw.update({'layout': x.layout, 'device': x.device})

    N = x.shape[axis]
    axis_idx = axis if axis >= 0 else (x.ndim + axis)
    padded_shape = (x.shape[:axis_idx] + (N + pad_left + pad_right,) +
                    x.shape[axis_idx + 1:])
    if out is None:
        out = backend.zeros(padded_shape, **kw)
    else:
        assert out.shape == padded_shape, (out.shape, padded_shape)
    if backend_name == 'tensorflow':  # TODO
        out = out.numpy()

    pad_right_idx = (out.shape[axis] -
                     (pad_right if pad_right != 0 else None))
    out[index_axis(pad_left, pad_right_idx, axis, x.ndim)] = x
    # out[pad_left:pad_right_idx] = x

    if pad_mode == 'zero':
        pass  # already done
    elif pad_mode == 'reflect':
        xflip = _flip(x, backend, axis, backend_name)
        out = _pad_reflect(x, xflip, out, pad_left, pad_right, N, axis)

    if backend_name == 'tensorflow':  # TODO
        out = backend.constant(out)
    return out


def _flip(x, backend, axis, backend_name='numpy'):
    if backend_name in ('numpy', 'tensorflow'):
        return x[flip_axis(axis, x.ndim)]
    elif backend_name == 'torch':
        axis = axis if axis >= 0 else (x.ndim + axis)
        return backend.flip(x, (axis,))


def pad_fast(x, pad_left, pad_right, out):
    N = x.shape[-1]
    out_len = out.shape[-1]

    # right
    xflip = x[::-1]
    right_idx = pad_left + N
    # steps = (out_len - right_idx) // N
    steps = np.ceil(pad_right / N)
    step = 2
    # print(steps, pad_right, N, right_idx, len(out))
    while step < steps + 2:
        i = step - 2
        start = right_idx + i*N
        # realized_len = len(out[start:start + N])
        realized_len = out_len - start
        # print(step, i, start, start + N, realized_len)
        if step % 2 == 0:
            out[..., start:start + N] = xflip[..., :realized_len]
        else:
            out[..., start:start + N] = x[..., :realized_len]
        step += 1

    # left
    xflip = x[::-1]
    left_idx = pad_left
    steps = np.ceil(pad_left / N)
    step = 2
    while step < steps + 2:
        end = left_idx
        i = step - 2
        realized_len = end - N  # target shortest quantity
        if step % 2 == 0:
            out[..., end - N:end] = xflip[..., -realized_len:]
        else:
            out[..., end - N:end] = x[..., -realized_len:]
        step += 1
    return out


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
        ix0, ix1 = idx(start, end), idx(-(max_step + 1), -1)
        if step % 2 == 0:
            out[ix0] = xflip[ix1]
        else:
            out[ix0] = x[ix1]
        step += 1
        already_stepped += max_step

    # pad right ##############################################################
    step, already_stepped  = 0, 0
    while already_stepped < pad_right:
        max_step = min(N - 1, pad_right - already_stepped)
        start = N + pad_left + already_stepped
        end = start + max_step
        ix0, ix1 = idx(start, end), idx(1, max_step + 1)
        if step % 2 == 0:
            out[ix0] = xflip[ix1]
        else:
            out[ix0] = x[ix1]
        step += 1
        already_stepped += max_step
    return out


def conj_reflections(backend, x, ind_start, ind_end):
    is_numpy = bool(type(backend).__module__ == 'numpy')
    N = ind_end - ind_start
    Np = x.shape[-1]

    # right
    start = ind_end
    end = start + N
    reflected = True
    while True:
        if reflected:
            if is_numpy:
                backend.conj(x[..., start:end], inplace=True)
            else:  # TODO any faster solution?
                x[..., start:end] = backend.conj(x[..., start:end])
        if end >= Np:
            break
        start += N
        end = min(start + N, Np)
        reflected = not reflected

    # left
    end = ind_start
    start = end - N
    reflected = True
    while True:
        if reflected:
            if is_numpy:
                backend.conj(x[..., start:end], inplace=True)
            else:
                x[..., start:end] = backend.conj(x[..., start:end])
        if start <= 0:
            break
        end -= N
        start = max(end - N, 0)
        reflected = not reflected


def _idx(a, b):
    return index_axis(a, b, -1, 1)

def pad2(x, out):
    N = len(x)
    ix0, ix1, ix2, ix3 = _idx(0, N), _idx(N, 2*N), _idx(2*N, 3*N), _idx(3*N, 4*N)
    ix4 = _idx(4*N, None)
    xflip = x[flip_axis(-1, 1)]
    out[ix0] = x
    out[ix1] = xflip
    out[ix2] = x
    out[ix3] = xflip
    out[ix4] = x

## helpers ###################################################################
def index_axis(i0, i1, axis, ndim, step=1):
    if axis < 0:
        slc = (slice(None),) * (ndim + axis) + (slice(i0, i1, step),)
    else:
        slc = (slice(None),) * axis + (slice(i0, i1, step),)
    return slc


def flip_axis(axis, ndim, step=1):
    return index_axis(-1, None, axis, ndim, -step)


def stride_axis(step, axis, ndim):
    return index_axis(None, None, axis, ndim, step)
