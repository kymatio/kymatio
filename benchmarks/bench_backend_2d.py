import kymatio.scattering2d.backend as backend

import torch

B = 128
N = 256

trials = 10
repeats = 3

x = torch.randn(B, 1, N, N, 2)
y = torch.randn(N, N, 2)

x = x.cuda()
y = y.cuda()

x = x.contiguous()
y = y.contiguous()

subsample_factors = (8, 16, 32, 64)

cdgmm = backend.cdgmm
modulus = backend.Modulus()
subsample = backend.SubsampleFourier()

name_list = ['cdgmm', 'modulus']
func_list = [cdgmm, modulus]
args_list = [(x, y), (x,)]
params_list = [(), ()]

for subsample_factor in subsample_factors:
    name_list.append('subsample')
    func_list.append(subsample)
    args_list.append((x,))
    params_list.append((subsample_factor,))

for name, func, args, params in zip(name_list, func_list, args_list, params_list):
    out = func(*(args + params))

    n_bytes = sum((x.nelement()*x.element_size() for x in (out,) + args))

    tm = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(trials):
            out = func(*(args + params))
        stop.record()

        stop.synchronize()
        tm.append(1e-3*start.elapsed_time(stop)/trials)

    tm = min(tm)

    #n_bytes = 2*x.nelement()*x.element_size() + y.nelement()*y.element_size()

    bw_gbs = n_bytes/tm/1e9

    display_name = name
    if params:
        display_name += '(' + ', '.join(str(x) for x in params) + ')'

    print('{: <20s}{:<5.4g} GB/s'.format(display_name + ':', bw_gbs))
