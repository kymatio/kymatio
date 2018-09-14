import torch
from torch.autograd import Variable
from scattering import Scattering1D
import math
from sklearn.utils import check_random_state

# Signal-related tests


def test_simple_scatterings(random_state=42):
    """
    Checks the behaviour of the scattering on simple signals
    (zero, constant, pure cosine)
    """
    rng = check_random_state(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(T, J, Q, normalize='l1')
    # zero signal
    x0 = Variable(torch.zeros(128, 1, T))
    s = scattering.forward(x0).data
    # check that s is zero!
    assert torch.max(torch.abs(s)) < 1e-7

    # constant signal
    x1 = Variable(rng.randn(1)[0] * torch.ones(1, 1, T))
    s1 = scattering.forward(x1).data
    # check that all orders above 1 are 0
    assert torch.max(torch.abs(s1[:, 1:])) < 1e-7

    # sinusoid scattering
    coords = Scattering1D.compute_meta_scattering(J, Q, order2=True)
    for _ in range(50):
        k = rng.randint(1, T // 2, 1)[0]
        x2 = torch.cos(2 * math.pi * float(k) * torch.arange(0, T) / float(T))
        x2 = Variable(x2.unsqueeze(0).unsqueeze(0))
        s2 = scattering.forward(x2).data

        for cc in coords.keys():
            if coords[cc]['order'] in ['0', '2']:
                assert torch.max(torch.abs(s2[:, cc])) < 1e-2


# Technical tests

def test_scattering_GPU_CPU(random_state=42, test_cuda=None):
    """
    This function tests whether the CPU computations are equivalent to
    the GPU ones

    Note that we can only achieve 1e-4 absolute l_infty error between GPU
    and CPU
    """
    torch.manual_seed(random_state)
    if test_cuda is None:
        test_cuda = torch.cuda.is_available()
    J = 6
    Q = 8
    T = 2**12

    if test_cuda:
        # build the scattering
        scattering = Scattering1D(T, J, Q)

        x = Variable(torch.randn(128, 1, T))
        s_cpu = scattering.forward(x).data

        scattering = scattering.cuda()
        x_gpu = Variable(x.data.clone().cuda())
        s_gpu = scattering.forward(x_gpu).data.cpu()
        # compute the distance
        assert torch.max(torch.abs(s_cpu - s_gpu)) < 1e-4


def test_coordinates(random_state=42):
    """
    Tests whether the coordinates correspond to the actual values (obtained
    with compute_meta_scattering), and with the vectorization
    """
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(T, J, Q, order2=True)
    x = Variable(torch.randn(128, 1, T))
    s_dico = scattering.forward(x, vectorize=False)
    s_dico = {k: s_dico[k].data for k in s_dico.keys()}
    s_vec = scattering.forward(x, vectorize=True).data

    coords = Scattering1D.compute_meta_scattering(J, Q, order2=True)

    assert len(s_dico) == s_vec.shape[1]

    for cc in range(s_vec.shape[1]):
        if coords[cc]['order'] == '0':
            k = 0
        elif coords[cc]['order'] == '1':
            # in this case, look for the key
            k = coords[cc]['key']
        elif coords[cc]['order'] == '2':
            k = (coords[cc]['key1'][0], coords[cc]['key1'][1],
                 coords[cc]['key2'][0], coords[cc]['key2'][1])
        else:
            assert False  # this case should NOT occur
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
    scattering = Scattering1D(T, J, Q)
    x = Variable(torch.randn(128, 1, T))
    for order2 in [True, False]:
        s_dico = scattering.forward(x, vectorize=False, order2=order2)
        for detail in [True, False]:
            # get the size of scattering
            size = scattering.precompute_size_scattering(
                J, Q, order2=order2, detail=detail)
            if detail:
                num_orders = {0: 0, 1: 0, 2: 0}
                for k in s_dico.keys():
                    if isinstance(k, int):
                        num_orders[0] += 1
                    else:
                        if len(k) == 2:  # order1
                            num_orders[1] += 1
                        elif len(k) == 4:
                            num_orders[2] += 1
                todo = 2 if order2 else 1
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
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(T, J, Q)
    x = Variable(torch.randn(128, 1, T), requires_grad=True)
    s = scattering.forward(x)
    loss = torch.sum(torch.abs(s))
    loss.backward()
    assert torch.max(torch.abs(x.grad.data)) > 0.
