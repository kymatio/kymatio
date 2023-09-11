"""
Classification on CIFAR10
=========================

Based on pytorch example for MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import argparse


class Scattering2dCNN(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG
    '''
    def __init__(self, in_channels, classifier_type='cnn'):
        super(Scattering2dCNN, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        if self.classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    self.in_channels = v

            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier =  nn.Linear(1024*4, 10)

        elif self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                        nn.Linear(self.K*8*8, 1024), nn.ReLU(),
                        nn.Linear(1024, 1024), nn.ReLU(),
                        nn.Linear(1024, 10))
            self.features = None

        elif self.classifier_type == 'linear':
            self.classifier = nn.Linear(self.K*8*8,10)
            self.features = None


    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



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

if __name__ == '__main__':

    """Train a simple Hybrid Scattering + CNN model on CIFAR.

        Three models are demoed:
        'linear' - scattering + linear model
        'mlp' - scattering + MLP
        'cnn' - scattering + CNN

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
        The model achieves around 88% testing accuracy after 10 epochs.

        scatter 1st order + linear achieves 64% in 90 epochs
        scatter 2nd order + linear achieves 70.5% in 90 epochs

        scatter + cnn achieves 88% in 15 epochs

    """
    parser = argparse.ArgumentParser(description='MNIST scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='cnn',help='classifier model')
    args = parser.parse_args()
    assert(args.classifier in ['linear','mlp','cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(32, 32), max_order=1)
        K = 17*3
    else:
        scattering = Scattering2D(J=2, shape=(32, 32))
        K = 81*3
    scattering = scattering.to(device)




    model = Scattering2dCNN(K,args.classifier).to(device)

    # DataLoaders
    num_workers = 4
    if use_cuda:
        pin_memory = True
    else:
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
    lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    for epoch in range(0, 90):
        train(model, device, train_loader, optimizer, epoch+1, scattering)
        scheduler.step()
        test(model, device, test_loader, scattering)
