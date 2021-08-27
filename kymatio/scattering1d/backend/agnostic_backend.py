import math
from ...toolkit import _infer_backend, get_kymatio_backend


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
    pad_right_idx = (out.shape[axis] -
                     (pad_right if pad_right != 0 else None))
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


def conj_reflections(backend, x, ind_start, ind_end, k, N, pad_left, pad_right,
                     trim_tm):
    """Conjugate reflections to mirror spins rather than negate them.

    Won't conjugate *at* boundaries:

        [..., 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        [..., 1, 2, 3, 4, 5, 6, 7,-6,-5,-4,-3,-2, 1, 2, 3, 4, 5, 6, 7]
    """
    import numpy as np


    # compute boundary indices from simulated reflected ramp at original length
    r = np.arange(N)
    rpo = np.pad(r, [pad_left, pad_right], mode='reflect')
    if trim_tm > 0:
        rpo = unpad_dyadic(rpo, N, len(rpo), len(rpo) // 2**trim_tm)

    # will conjugate where sign is negative
    rpdiffo = np.diff(rpo)
    rpdiffo = np.hstack([rpdiffo[0], rpdiffo])
    # mark rising ramp as +1, including bounds, and everything else as -1;
    # -1 will conjugate. This instructs to not conjugate: non-reflections, bounds.
    # diff will mark endpoints with sign opposite to rise's; correct manually
    rpdiffo[np.where(np.diff(rpdiffo) > 0)[0]] = 1
    rpdiff = rpdiffo[::2**k]

    idxs = np.where(rpdiff == -1)[0]
    # conjugate to left of left bound
    assert ind_start - 1 in idxs, (ind_start - 1, idxs)
    # do not conjugate left bound
    assert ind_start not in idxs, (ind_start, idxs)
    # conjugate to right of right bound
    # (Python indexing excludes `ind_end`, so `ind_end - 1` is the right bound)
    assert ind_end in idxs, (ind_end, idxs)
    # do not conjugate the right bound
    assert ind_end - 1 not in idxs, (ind_end - 1, idxs)

    # infer & fetch backend
    backend_name = type(x).__module__.lower().split('.')[0]
    B = get_kymatio_backend(backend_name)

    # conjugate and assign
    if (backend_name in ('numpy', 'tensorflow') or
            (backend_name == 'torch' and getattr(x, 'requires_grad', False))):
        ic = [0, *(np.where(np.diff(idxs) > 1)[0] + 1)]
        ic.append(None)
        ic = np.array(ic)
        slices_contiguous = []
        for i in range(len(ic) - 1):
            s, e = ic[i], ic[i + 1]
            start = idxs[s]
            end = idxs[e - 1] + 1 if e is not None else None
            slices_contiguous.append(slice(start, end))

        inplace = bool(backend_name == 'numpy' or
                       (backend_name == 'torch' and
                        not getattr(x, 'requires_grad', False)))
        for slc in slices_contiguous:
            x = B.assign_slice(x, B.conj(x[..., slc], inplace=inplace),
                               index_axis_with_array(slc, axis=-1, ndim=x.ndim))
    else:  # TODO any faster solution?
        x = B.assign_slice(x, B.conj(x[..., idxs]),
                           index_axis_with_array(idxs, axis=-1, ndim=x.ndim))
    # only because of tensorflow
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
