from tqdm import tqdm
import math
import torch
import torch.optim
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torchnet.engine import Engine
from torch.autograd import Variable
import torch.nn.functional as F
from scatwave.scattering import Scattering


def get_iterator(mode):
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=128, num_workers=4, shuffle=mode)


def conv_init(ni, no, k):
    return torch.Tensor(no, ni, k, k).normal_(0, 2/math.sqrt(ni*k*k))


def linear_init(ni, no):
    return torch.Tensor(no, ni).normal_(0, 2/math.sqrt(ni))


def f(o, params, stats, mode):
    o = F.batch_norm(o, running_mean=stats['bn.running_mean'],
                     running_var=stats['bn.running_var'],
                     weight=params['bn.weight'],
                     bias=params['bn.bias'], training=mode)
    o = F.conv2d(o, params['conv1.weight'], params['conv1.bias'])
    o = F.relu(o)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['linear2.weight'], params['linear2.bias'])
    o = F.relu(o)
    o = F.linear(o, params['linear3.weight'], params['linear3.bias'])
    return o


def main():
    """Train a simple Hybrid Scattering + CNN model on MNIST.

    Scattering features are normalized by batch normalization.
    The model achieves 99.6% testing accuracy after 10 epochs.
    """
    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)

    scat = Scattering(M=28, N=28, J=2).cuda()
    K = 81

    params = {
        'conv1.weight':     conv_init(K, 64, 1),
        'conv1.bias':       torch.zeros(64),
        'bn.weight':        torch.Tensor(K).uniform_(),
        'bn.bias':          torch.zeros(K),
        'linear2.weight':   linear_init(64*7*7, 512),
        'linear2.bias':     torch.zeros(512),
        'linear3.weight':   linear_init(512, 10),
        'linear3.bias':     torch.zeros(10),
    }

    stats = {'bn.running_mean': torch.zeros(K).cuda(),
             'bn.running_var': torch.ones(K).cuda()}

    for k, v in params.items():
        params[k] = Variable(v.cuda(), requires_grad=True)

    def h(sample):
        x = scat(sample[0].float().cuda().unsqueeze(1) / 255.0).squeeze(1)
        inputs = Variable(x)
        targets = Variable(torch.LongTensor(sample[1]).cuda())
        o = f(inputs, params, stats, sample[2])
        return F.cross_entropy(o, targets), o

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        classerr.reset()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print 'Training accuracy:', classerr.value()

    def on_end(state):
        print 'Training' if state['train'] else 'Testing', 'accuracy'
        print classerr.value()

    optimizer = torch.optim.SGD(params.values(), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end'] = on_end
    print 'Training:'
    engine.train(h, get_iterator(True), 10, optimizer)
    print 'Testing:'
    engine.test(h, get_iterator(False))


if __name__ == '__main__':
    main()
