""" This script will test the submodules used by the scattering module"""
import torch
import numpy as np
import os.path
from scattering import Scattering3D
from scattering.scattering3d.utils import compute_integrals


def relative_difference(a, b):
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))


def test_against_standard_computations():
    file_path = os.path.abspath(os.path.dirname(__file__))
    data = torch.load(os.path.join(file_path, 'data/ref_pyscatharm.pt'))
    x = data['x']
    scat_ref = data['Sx']
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]

    N, O = M, M
    sigma = 1

    scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    x_cpu = x.cpu()

    order_0_cpu = compute_integrals(x_cpu, integral_powers)
    order_1_cpu, order_2_cpu = scattering(x_cpu, order_2=True,
                            method='integral', integral_powers=integral_powers)


    order_0_cpu = order_0_cpu.numpy().reshape((batch_size, -1))
    start = 0
    end = order_0_cpu.shape[1]
    order_0_ref = scat_ref[:,start:end].numpy()

    order_1_cpu = order_1_cpu.numpy().reshape((batch_size, -1))
    start = end
    end += order_1_cpu.shape[1]
    order_1_ref = scat_ref[:, start:end].numpy()

    order_2_cpu = order_2_cpu.numpy().reshape((batch_size, -1))
    start = end
    end += order_2_cpu.shape[1]
    order_2_ref = scat_ref[:, start:end].numpy()

    order_0_diff_cpu = relative_difference(order_0_ref, order_0_cpu)
    order_1_diff_cpu = relative_difference(order_1_ref, order_1_cpu)
    order_2_diff_cpu = relative_difference(order_2_ref, order_2_cpu)

    x_gpu = x_cpu.cuda()
    order_0_gpu = compute_integrals(x_gpu, integral_powers)
    order_1_gpu, order_2_gpu = scattering(x_gpu, order_2=True,
                            method='integral', integral_powers=integral_powers)
    order_0_gpu = order_0_gpu.cpu().numpy().reshape((batch_size, -1))
    order_1_gpu = order_1_gpu.cpu().numpy().reshape((batch_size, -1))
    order_2_gpu = order_2_gpu.cpu().numpy().reshape((batch_size, -1))

    order_0_diff_gpu = relative_difference(order_0_ref, order_0_gpu)
    order_1_diff_gpu = relative_difference(order_1_ref, order_1_gpu)
    order_2_diff_gpu = relative_difference(order_2_ref, order_2_gpu)

    assert  order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert  order_1_diff_cpu < 1e-6, "CPU : order 1 do not match, diff={}".format(order_1_diff_cpu)
    assert  order_2_diff_cpu < 1e-6, "CPU : order 2 do not match, diff={}".format(order_2_diff_cpu)

    assert  order_0_diff_gpu < 1e-6, "GPU : order 0 do not match, diff={}".format(order_0_diff_gpu)
    assert  order_1_diff_gpu < 1e-6, "GPU : order 1 do not match, diff={}".format(order_1_diff_gpu)
    assert  order_2_diff_gpu < 1e-6, "GPU : order 2 do not match, diff={}".format(order_2_diff_gpu)
