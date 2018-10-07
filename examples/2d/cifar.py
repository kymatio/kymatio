"""
Self-contained classification example on MNIST

Based on pytorch example for MNIST
"""


import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from scattering import Scattering2D
import torch
import argparse


import math

import torch.nn as nn
import torch.nn.init as init


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__(K)
                    16,          8           4                2
         cfg = [128,128, 'M', 256, 256, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_channels = K
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def make_layers():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M', 1024, 1024, 1024, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_perm(layer_to_perm=7,fixed_order=True,poolstride=False):
    #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 0   1    2    3   4     5     6   7   8    9     10 11    12   13   14   15   16   17
    if layer_to_perm < 2:
        sz = 32
    elif layer_to_perm < 5:
        sz = 16
    elif layer_to_perm < 9:
        sz = 8
    elif layer_to_perm < 13:
        sz = 4
    else:
        sz = 2
    if not fixed_order:
        sz = -1
    assert(layer_to_perm<17)
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers_perm(cfg['D'], batch_norm=True,layer_to_perm=layer_to_perm,sz=sz,poolstride=poolstride))

def vgg16_avg_perm(layer_to_perm=7,fixed_order=True,poolstride=False):
    #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 0   1    2    3   4     5     6   7   8    9     10 11    12   13   14   15   16   17
    if layer_to_perm < 2:
        sz = 32
    elif layer_to_perm < 5:
        sz = 16
    elif layer_to_perm < 9:
        sz = 8
    elif layer_to_perm < 13:
        sz = 4
    else:
        sz = 2
    if not fixed_order:
        sz = -1
    assert(layer_to_perm<17)
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_avg(make_layers_perm(cfg['F'][:-1], batch_norm=True,layer_to_perm=layer_to_perm,sz=sz,poolstride=poolstride))

def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

def train(model, device, train_loader, optimizer, epoch, scat):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scat(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, scat):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scat(data))
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """Train a simple Hybrid Scattering + CNN model on MNIST.

        Three models are demoed:
        'linear' - scattering + linear model
        'mlp' - scattering + MLP
        'cnn' - scattering + CNN

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
        The model achieves XX testing accuracy after 10 epochs.
    """
    parser = argparse.ArgumentParser(description='MNIST scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=2,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='linear',help='classifier model')
    args = parser.parse_args()
    assert(args.classifier in ['linear','mlp','cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    scat = Scattering2D(M=28, N=28, J=2)
    if use_cuda:
        scat = scat.cuda()

    K = 81


    if args.classifier == 'cnn':
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 32, 3,padding=1), nn.ReLU(),
            nn.Conv2d(K, 32, 3,padding=1), nn.ReLU(),
            View(64*7*7),
            nn.Linear(32 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)

    elif args.classifier == 'mlp':
        model = nn.Sequential(
            View(K, 8, 8),
            nn.BatchNorm2d(K),
            View(K*8*8),
            nn.Linear(K*7*7, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    elif args.classifier == 'linear':
        model = nn.Sequential(
           # View(K, 7, 7),
            #nn.BatchNorm2d(K),
            View(K * 7 * 7),
            nn.Linear(K * 7 * 7, 10)
        )
    else:
        ValueError()

    model.to(device)

    # DataLoaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, **kwargs)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)

    for epoch in range(1, 16):
        train( model, device, train_loader, optimizer, epoch, scat)
        test(model, device, test_loader, scat)


if __name__ == '__main__':
    main()
