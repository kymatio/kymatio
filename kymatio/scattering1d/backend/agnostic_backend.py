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


def conj_reflections(backend, x, ind_start, ind_end, k, N, pad_left, pad_right):
    """Conjugate reflections to mirror spins rather than negate them.

    Won't conjugate *at* boundaries:

        [..., 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        [..., 1, 2, 3, 4, 5, 6, 7,-6,-5,-4,-3,-2, 1, 2, 3, 4, 5, 6, 7]
    """
    import numpy as np

    is_numpy = bool(type(backend).__module__ == 'numpy')
    # N = ind_end - ind_start
    Np = x.shape[-1]

    # compute boundary indices from simulated reflected ramp at original length
    r = np.arange(N)
    rpo = np.pad(r, [pad_left, pad_right], mode='reflect')
    rp = rpo[::2**k]
    # will conjugate where sign is negative
    rpdiffo = np.diff(rpo)
    rpdiffo = np.hstack([rpdiffo[0], rpdiffo])
    # mark rising ramp as +1, including bounds, and everything else as -1;
    # -1 will conjugate. This instructs to not conjugate: non-reflections, bounds.
    # diff will mark endpoints with sign opposite to rise's; correct manually
    rpdiffo[np.where(np.diff(rpdiffo) > 0)[0]] = 1
    rpdiffo[np.where(np.diff(rpdiffo) < 0)[0] + 1] = 1


    rpdiff = rpdiffo[::2**k]

    # boundary indices
    rpdiffo_d = np.diff(rpdiffo)
    rpdiff_d  = np.diff(rpdiff)
    idxso = np.where(rpdiffo_d != 0)[0]
    idxs  = np.where(rpdiff_d  != 0)[0]

    # strided "boundary" is ambiguous by seeking against change in direction
    # of ramp; we disambiguate by defining per original bounds:
    #   "start" is first sample to right of where ramp stops falling
    #   "end" is sample at which ramp stops rising
    # but this is more a preferred perspective as it's a reformulation of
    # "change in direction"; what remains is to incorporate information on
    # location of original boundaries
    idxs_o_match = [np.where(idxs == io)[0] for io in idxso // 2**k]
    idxs_o_match = np.array([iom for iom in idxs_o_match if len(iom) > 0])
    # a positive diff(diff) indicates [fall -> rise], or left bound in original;
    # negative thus right bound. adjust left such that, if it doesn't land *at*
    # unsubsampled bound, we increment by 1 - and opposite for right.
    for i, idx in enumerate(idxs):
        assert rpdiff_d[idx] != 0, (idx, rpdiff_d, rpdiff_d[idx])
        if rpdiff_d[idx] > 0 and idx not in idxs_o_match:
            idxs[i] += 1
        elif rpdiff_d[idx] < 0 and idx in idxs_o_match:
            idxs[i] += 1



    try:
        assert ind_start   in idxs, (ind_start,   idxs)
        # `ind_end` indexes *beyond* the bound to *include* it in unpadding
        # per Pythonic indexing (i.e. x[start:end], end is excluded),
        # whereas `idxs` are placed *at* bounds.
        assert ind_end - 1 in idxs, (ind_end - 1, idxs)
    except:
        1/0
    if k > 1:
        a0, b0 = ind_start - 3, ind_start + 3
        a1, b1 = ind_end   - 3 - 1, ind_end   + 3 - 1
        print(rp[a0:b0])
        print(rpdiff[a0:b0])
        print(rp[a1:b1])
        print(rpdiff[a1:b1])
        print()


    idxs_right = idxs[idxs >= ind_end]
    idxs_left  = idxs[idxs <= ind_start]

    # right
    start = idxs_right[0] + 1  # don't conjugate at boundaries
    end   = (idxs_right[1] if len(idxs_right) >= 2 else
             Np)
    i = 2
    reflected = True
    while True:
        if reflected:
            if is_numpy:
                backend.conj(x[..., start:end], inplace=True)
            else:  # TODO any faster solution?
                x[..., start:end] = backend.conj(x[..., start:end])
        # if end >= Np:
        #     break
        # 'reflect' adds N - 1 samples rather than N (the boundary sample skipped)
        start = idxs_right[i] + 1
        end   = (idxs_right[i + 1] if len(idxs_right) >= i + 2 else
                 Np)
        i += 2
        # start += (N - 1)
        # end = min(start + (N - 1), Np)
        reflected = not reflected
        if i > len(idxs_right) - 1:
            break

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
