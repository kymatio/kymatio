"""
Demo code for the module
"""
import torch
from torch.autograd import Variable
from time import time
from scatwave import Scattering1D


def mydemo(T, J, Q, batch_size=8, order2=False, cuda=False):
    # define the input
    tic = time()
    x = torch.randn(batch_size, 1, T)
    if cuda:
        print('******GPU******')
        x = x.cuda()
    else:
        print('******CPU******')
    x = Variable(x)
    print('Input generated in {0:.3f} sec.'.format(time() - tic))
    # initialize the scattering
    tic = time()
    scattering = Scattering1D(T, J, Q, order2=order2)
    # scat.set_default_args(order2=order2)
    if cuda:
        scattering = scattering.cuda()
    print('Filters initialized in {0:.3f} sec.'.format(time() - tic))
    # Forward it
    tic = time()
    S = scattering.forward(x)
    print('Scattering computed in {0:.3f} sec.'.format(time() - tic))


if __name__ == '__main__':
    T = 2**11  # 2048
    J = 5
    Q = 6
    batch_size = 8
    order2 = False
    mydemo(T, J, Q, batch_size=batch_size, order2=order2, cuda=False)
    if torch.cuda.is_available():
        mydemo(T, J, Q, batch_size=batch_size, order2=order2, cuda=True)
