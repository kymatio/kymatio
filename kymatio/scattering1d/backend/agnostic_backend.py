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


def conj_reflections(backend, x, ind_start, ind_end, k, N, pad_left, pad_right,
                     trim_tm):
    """Conjugate reflections to mirror spins rather than negate them.

    Won't conjugate *at* boundaries:

        [..., 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        [..., 1, 2, 3, 4, 5, 6, 7,-6,-5,-4,-3,-2, 1, 2, 3, 4, 5, 6, 7]
    """
    slices_contiguous = _emulate_get_conjugation_indices(
        N, k, pad_left, pad_right, trim_tm)

    def is_in(ind):
        for slc in slices_contiguous:
            if ind in (slc.start, slc.stop):
                return True
        return False

    # conjugate to left of left bound
    assert is_in(ind_start), (ind_start - 1, slices_contiguous)
    # do not conjugate left bound
    assert not is_in(ind_start + 1), (ind_start, slices_contiguous)
    # conjugate to right of right bound
    # (Python indexing excludes `ind_end`, so `ind_end - 1` is the right bound)
    assert is_in(ind_end), (ind_end, slices_contiguous)
    # do not conjugate the right bound
    assert not is_in(ind_end - 1), (ind_end - 1, slices_contiguous)

    # infer & fetch backend
    backend_name = type(x).__module__.lower().split('.')[0]
    B = get_kymatio_backend(backend_name)

    # conjugate and assign
    if backend_name == 'torch':
        import torch
        torch110 = bool(int(torch.__version__.split('.')[1]) >= 10)
        inplace = bool(torch110 or not getattr(x, 'requires_grad', False))
    else:
        inplace = bool(backend_name == 'numpy')
    for slc in slices_contiguous:
        x = B.assign_slice(x, B.conj(x[..., slc], inplace=inplace),
                           index_axis_with_array(slc, axis=-1, ndim=x.ndim))
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


def _emulate_get_conjugation_indices(N, K, pad_left, pad_right, trim_tm):
    import numpy as np

    orig_dyadic_len = int(2**np.ceil(np.log2(N)))
    padded_len = N + pad_left + pad_right
    pad_factor = int(np.log2(padded_len / orig_dyadic_len))
    pad_factor -= trim_tm

    padded_len = 2**pad_factor * int(2**np.ceil(np.log2(N)))
    pad_left = int(np.ceil((padded_len - N) / 2))
    pad_right = padded_len - pad_left - N

    ramp_len = N
    ramp_padded_len = ramp_len + pad_left + pad_right
    ramp_orig_left = pad_left
    ramp_orig_right = ramp_orig_left + (N - 1)

    # left
    border_idxs_left = []
    idx = ramp_orig_left
    last_is_start = False
    while idx > 0:
        border_idxs_left.append(idx)
        idx -= (N - 1)
        last_is_start = not last_is_start
    # right
    border_idxs_right = []
    idx = ramp_orig_right
    while idx < ramp_padded_len:
        border_idxs_right.append(idx)
        idx += (N - 1)
    # concat
    border_idxs_left, border_idxs_right = [np.array(b) for b in
                                           (border_idxs_left, border_idxs_right)]
    border_idxs = np.array([*border_idxs_left[::-1], *border_idxs_right])

    # given border indices, and status of initial index, will find all
    # conjugation indices
    ixs = []
    # we're at down ramp (and to left of it, even if at last sample of down)
    start_of_down = last_is_start
    # if `border_idxs[0]` is at down ramp, include all points up to that
    # border index
    if start_of_down:
        ixs.append(0)
        # no longer at start
        start_of_down = False
        # use all indices
        bidxs = border_idxs
    else:
        ix = border_idxs[0] + 1
        ix = np.ceil(ix / 2**K)
        ix_inclusive_orig = 2**K * ix
        if ix_inclusive_orig < ramp_padded_len:
            ixs.append(ix)
        # still no longer at start, but we don't begin at `0`
        start_of_down = False
        # drop first since already used
        bidxs = border_idxs[1:]

    for i, ix in enumerate(bidxs):
        # idx at down ramp will be `1` right or left of border idx
        ix_inclusive = ((ix + 1) if start_of_down else
                        (ix - 1))

        # down ramp index in strided space
        iis = ix_inclusive / 2**K
        ix_inclusive_strided = (np.ceil(iis) if start_of_down else
                                np.floor(iis))
        # project back to original
        ix_inclusive_orig = 2**K * ix_inclusive_strided

        if ix_inclusive_orig < ramp_padded_len:
            ixs.append(ix_inclusive_strided)
            # this assumes strided step size is smaller than reflection size
            start_of_down = not start_of_down
        # move onto next

    # if we ended left, add finishing index
    if not start_of_down and len(ixs) > 0:
        ix = ramp_padded_len // 2**K - 1
        ixs.append(ramp_padded_len // 2**K - 1)
    ixs = np.array(ixs).astype(int)

    assert len(ixs) % 2 == 0, len(ixs)

    slices_contiguous = []
    for i in range(0, len(ixs), 2):
        s, e = ixs[i], ixs[i + 1] + 1
        slices_contiguous.append(slice(s, e))
    out = slices_contiguous

    return out
