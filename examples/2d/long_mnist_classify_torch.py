"""
Classification of Few Sample MNIST with Scattering
=====================================================================
Here we demonstrate a simple application of scattering on the MNIST dataset.
We use 5000 MNIST samples to train a linear classifier. Features are normalized by batch normalization.
Please also see more extensive classification examples/2d/cifar.py
that consider training CNNs on top of the scattering. These are
are included as  executable script with command line arguments
"""


###############################################################################
# If a GPU is available, let's use it!
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

###############################################################################
# For reproducibility, we fix the seed of the random number generator.
from numpy.random import RandomState
import numpy as np
torch.manual_seed(42)
prng = RandomState(42)

###############################################
# Create dataloaders
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets

num_workers = 4
if use_cuda:
    pin_memory = True
else:
    pin_memory = False

train_data = datasets.MNIST(
                scattering_datasets.get_dataset_dir('MNIST'),
                train=True, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                ]))

# Extract a subset of 5000 samples from MNIST training
random_permute=prng.permutation(np.arange(0,60000))[0:5000]
train_data.data = train_data.data[random_permute]
train_data.targets = train_data.targets[random_permute]
train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

# Create the test loader on the full MNIST test set
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        scattering_datasets.get_dataset_dir('MNIST'),
        train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


###############################################################################
# This will help us define networks a bit more cleanly
import torch.nn as nn
class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)


###############################################################################
# Create a training and test function
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, scattering):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)
############################################################################
# Train a simple Hybrid Scattering + CNN model on MNIST.

from kymatio.torch import Scattering2D
import torch.optim
import math


# Evaluate linear model on top of scattering
scattering = Scattering2D(shape = (28, 28), J=2).to(device)
K = 81 #Number of output coefficients for each spatial postiion


model = nn.Sequential(
    View(K, 7, 7),
    nn.BatchNorm2d(K),
    View(K * 7 * 7),
    nn.Linear(K * 7 * 7, 10)
).to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            weight_decay=0.0005)
for epoch in range(0, 20):
    train( model, device, train_loader, optimizer, scattering)

acc = test(model, device, test_loader, scattering)
print('Scattering order 2 linear model test accuracy: %.2f' % acc)
