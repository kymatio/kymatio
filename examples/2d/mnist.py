"""
Classification of handwritten digits
====================================

Based on pytorch example for MNIST
"""


import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import kymatio.datasets as scattering_datasets
import kymatio
import torch
import argparse
import math

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, scattering):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
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

        scatter + linear achieves 99.15% in 15 epochs
        scatter + cnn achieves 99.3% in 15 epochs

    """
    parser = argparse.ArgumentParser(description='MNIST scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=2,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='linear',help='classifier model')
    args = parser.parse_args()
    assert(args.classifier in ['linear','mlp','cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(28, 28), max_order=1)
        K = 17
    else:
        scattering = Scattering2D(J=2, shape=(28, 28))
        K = 81
    if use_cuda:
        scattering = scattering.cuda()




    if args.classifier == 'cnn':
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 64, 3,padding=1), nn.ReLU(),
            View(64*7*7),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)

    elif args.classifier == 'mlp':
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            View(K*7*7),
            nn.Linear(K*7*7, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    elif args.classifier == 'linear':
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            View(K * 7 * 7),
            nn.Linear(K * 7 * 7, 10)
        )
    else:
        raise ValueError('Classifier should be cnn/mlp/linear')

    model.to(device)

    #initialize
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, 2./math.sqrt(n))
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 2./math.sqrt(m.in_features))
            m.bias.data.zero_()

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)

    for epoch in range(1, 16):
        train( model, device, train_loader, optimizer, epoch, scattering)
        test(model, device, test_loader, scattering)


if __name__ == '__main__':
    main()
