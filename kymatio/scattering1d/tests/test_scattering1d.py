import torch
from kymatio import Scattering1D
import math
import os
import numpy as np
import kymatio.scattering1d.backend as backend
import warnings

# Signal-related tests

if backend.NAME == 'skcuda':
    force_gpu = True
else:
    force_gpu = False

def test_simple_scatterings(random_state=42):
    """
    Checks the behaviour of the scattering on simple signals
    (zero, constant, pure cosine)
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, normalize='l1')
    if force_gpu:
        scattering = scattering.cuda()
    # zero signal
    x0 = torch.zeros(128, 1, T)
    if force_gpu:
        x0 = x0.cuda()
    s = scattering.forward(x0)
    if force_gpu:
        s = s.cpu()
    # check that s is zero!
    assert torch.max(torch.abs(s)) < 1e-7

    # constant signal
    x1 = rng.randn(1)[0] * torch.ones(1, 1, T)
    if force_gpu:
        x1 = x1.cuda()
    s1 = scattering.forward(x1)
    if force_gpu:
        s1 = s1.cpu()
    # check that all orders above 1 are 0
    assert torch.max(torch.abs(s1[:, 1:])) < 1e-7

    # sinusoid scattering
    meta = scattering.meta()
    for _ in range(50):
        k = rng.randint(1, T // 2, 1)[0]
        x2 = torch.cos(2 * math.pi * float(k) * torch.arange(0, T, dtype=torch.float32) / float(T))
        x2 = x2.unsqueeze(0).unsqueeze(0)
        if force_gpu:
            x2 = x2.cuda()
        s2 = scattering.forward(x2)
        if force_gpu:
            s2 = s2.cpu()

        assert(s2[:,meta['order'] != 1,:].abs().max() < 1e-2)


def test_sample_scattering():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.
    """
    test_data_dir = os.path.dirname(__file__)
    test_data_filename = os.path.join(test_data_dir, 'test_data_1d.pt')
    data = torch.load(test_data_filename, map_location='cpu')

    x = data['x']
    J = data['J']
    Q = data['Q']
    Sx0 = data['Sx']

    T = x.shape[2]

    scattering = Scattering1D(J, T, Q)

    if force_gpu:
        scattering = scattering.cuda()
        x = x.cuda()

    Sx = scattering.forward(x)

    if force_gpu:
        Sx = Sx.cpu()

    assert (Sx - Sx0).abs().max() < 1e-6


def test_computation_Ux(random_state=42):
    """
    Checks the computation of the U transform (no averaging for 1st order)
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, normalize='l1', average=False,
                              max_order=1, vectorize=False)
    # random signal
    x = torch.from_numpy(rng.randn(1, 1, T)).float()

    if force_gpu:
        scattering.cuda()
        x = x.cuda()

    s = scattering.forward(x)

    # check that the keys in s correspond to the order 0 and second order
    for k in range(len(scattering.psi1_f)):
        assert (k,) in s.keys()
    for k in s.keys():
        if k is not ():
            assert k[0] < len(scattering.psi1_f)
        else:
            assert True


# Technical tests

def test_scattering_GPU_CPU(random_state=42, test_cuda=None):
    """
    This function tests whether the CPU computations are equivalent to
    the GPU ones

    Note that we can only achieve 1e-4 absolute l_infty error between GPU
    and CPU
    """
    if force_gpu:
        return

    torch.manual_seed(random_state)
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    J = 6
    Q = 8
    T = 2**12

    if test_cuda:
        # build the scattering
        scattering = Scattering1D(J, T, Q)

        x = torch.randn(128, 1, T)
        s_cpu = scattering.forward(x)

        scattering = scattering.cuda()
        x_gpu = x.clone().cuda()
        s_gpu = scattering.forward(x_gpu).cpu()
        # compute the distance
        assert torch.max(torch.abs(s_cpu - s_gpu)) < 1e-4


def test_coordinates(random_state=42):
    """
    Tests whether the coordinates correspond to the actual values (obtained
    with Scattering1d.meta()), and with the vectorization
    """
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, max_order=2)
    x = torch.randn(128, 1, T)

    if force_gpu:
        scattering.cuda()
        x = x.cuda()

    scattering.vectorize = False
    s_dico = scattering.forward(x)
    s_dico = {k: s_dico[k].data for k in s_dico.keys()}
    scattering.vectorize = True
    s_vec = scattering.forward(x)

    if force_gpu:
        s_dico = {k: s_dico[k].cpu() for k in s_dico.keys()}
        s_vec = s_vec.cpu()

    meta = scattering.meta()

    assert len(s_dico) == s_vec.shape[1]

    for cc in range(s_vec.shape[1]):
        k = meta['key'][cc]
        diff = s_vec[:, cc] - torch.squeeze(s_dico[k])
        assert torch.max(torch.abs(diff)) < 1e-7


def test_precompute_size_scattering(random_state=42):
    """
    Tests that precompute_size_scattering computes a size which corresponds
    to the actual scattering computed
    """
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, vectorize=False)
    x = torch.randn(128, 1, T)

    if force_gpu:
        scattering.cuda()
        x = x.cuda()

    for max_order in [1, 2]:
        scattering.max_order = max_order
        s_dico = scattering.forward(x)
        for detail in [True, False]:
            # get the size of scattering
            size = scattering.output_size(detail=detail)
            if detail:
                num_orders = {0: 0, 1: 0, 2: 0}
                for k in s_dico.keys():
                    if k is ():
                        num_orders[0] += 1
                    else:
                        if len(k) == 1:  # order1
                            num_orders[1] += 1
                        elif len(k) == 2:
                            num_orders[2] += 1
                todo = 2 if max_order == 2 else 1
                for i in range(todo):
                    assert num_orders[i] == size[i]
                    # check that the orders are completely equal
            else:
                assert len(s_dico) == size


def test_differentiability_scattering(random_state=42):
    """
    It simply tests whether it is really differentiable or not.
    This does NOT test whether the gradients are correct.
    """

    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now)."), RuntimeWarning, stacklevel=2)
        return

    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q)
    x = torch.randn(128, 1, T, requires_grad=True)

    if force_gpu:
        scattering.cuda()
        x = x.cuda()

    s = scattering.forward(x)
    loss = torch.sum(torch.abs(s))
    loss.backward()
    assert torch.max(torch.abs(x.grad)) > 0.
