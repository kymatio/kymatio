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


class Scat2dCNN(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG
    '''
    def __init__(self, in_channels, classifier_type='cnn'):
        super(Scat2dCNN, self).__init__()
        cfg = [128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512]
        layers = []
        self.K = in_channels
        self.bn = nn.BatchNorm2d(self.K)
        if classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    in_channels = v

            layers += [nn.AdaptiveAvgPool2d(1)]
            self.features = nn.Sequential(*layers)
            self.classifier =  nn.Linear(512, 10)

        elif classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                        nn.Linear(self.K*8*8, 1024), nn.ReLU(),
                        nn.Linear(1024, 1024), nn.ReLU(),
                        nn.Linear(1024, 10))
            self.features = None

        elif classifier_type == 'linear':
            self.classifier = nn.Linear(self.K*8*8,10)
            self.features = None



         # Initialize weights for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



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
    """Train a simple Hybrid Scattering + CNN model on CIFAR.

        Three models are demoed:
        'linear' - scattering + linear model
        'mlp' - scattering + MLP
        'cnn' - scattering + CNN

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
        The model achieves XX testing accuracy after 10 epochs.

        scatter + linear achieves 99.15% in 15 epochs
        scatter + cnn achieves 99.3% in 15 epochs

    """
    parser = argparse.ArgumentParser(description='MNIST scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='linear',help='classifier model')
    args = parser.parse_args()
    assert(args.classifier in ['linear','mlp','cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scat = Scattering2D(M=32, N=32, J=2,order2=False)
        K = 17*3
    else:
        scat = Scattering2D(M=32, N=32, J=2)
        K = 81*3
    if use_cuda:
        scat = scat.cuda()




    model = Scat2dCNN(K,args.classifier).to(device)

    # DataLoaders
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, **kwargs)

    # Optimizer
    lr = 0.1
    for epoch in range(0, 90):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2

        train(model, device, train_loader, optimizer, epoch+1, scat)
        test(model, device, test_loader, scat)



if __name__ == '__main__':
    main()
