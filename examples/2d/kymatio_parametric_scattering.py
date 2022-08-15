"""
Parametric Scattering in Kymatio
=======================================
Here we demonstrate how to develop a novel scattering utilizing the Kymatio
code base. We then train a linear classifier and visualize the filters.
"""

###############################################################################
# Preliminaries
# -------------
# Kymatio can be used to develop novel scattering algorithms. We'll use class
# inheritance to massively reduce the amount of code we have to write. 
# Furthermore, we'll use Torch as we want differentiability and optimizability.
from kymatio.torch import Scattering2D
import torch

###############################################################################
# We'll also import various libraries needed for implementation

import types

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


###############################################################################
# Implementation of parametric scattering - wavelet generation
# -----------------------------------------------------------------------------
# We wish to develop a version of the scattering transform where the wavelet
# parameters can be optimized. Currently, in Kymatio, the filter generation 
# is defined using Numpy. Numpy does not give us differentiablity, so we 
# reimplement the morlet generation utilizing Torch.

def morlets(grid_or_shape, theta, xis, sigmas, slants, morlet=True, ifftshift=True, fft=True, return_gabors=False):
    """Creates morlet wavelet filters from inputs

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            theta -- global orientations of the wavelets
            xis -- frequency scales of the wavelets
            sigmas -- gaussian window scales of the wavelets
            slants -- slants of the wavelets
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)
        Returns:
            wavelets -- the wavelet filters

    """
    orientations = torch.cat((torch.cos(theta).unsqueeze(1),torch.sin(theta).unsqueeze(1)), dim =1)
    n_filters, ndim = orientations.shape
    wave_vectors = orientations * xis[:, np.newaxis]
    _, _, gauss_directions = torch.linalg.svd(orientations[:, np.newaxis])
    gauss_directions = gauss_directions / sigmas[:, np.newaxis, np.newaxis]
    indicator = torch.arange(ndim, device=slants.device) < 1
    slant_modifications = (1.0 * indicator + slants[:, np.newaxis] * ~indicator)
    gauss_directions = gauss_directions * slant_modifications[:, :, np.newaxis]
    if return_gabors:
        wavelets, gabors = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft, return_gabors=True)
    else:
        wavelets = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft)

    norm_factors = (2 * 3.1415 * sigmas * sigmas / slants).unsqueeze(1)

    if type(grid_or_shape) == tuple:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape[0]]).unsqueeze(2).repeat(1,1,grid_or_shape[1])
    else:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape.shape[1]]).unsqueeze(2).repeat(1,1,grid_or_shape.shape[2])

    wavelets = wavelets / norm_factors
    if return_gabors:
      return gabors/norm_factors

    return wavelets


def raw_morlets(grid_or_shape, wave_vectors, gaussian_bases, morlet=True, ifftshift=True, fft=True, return_gabors=False):
    """ Helper function for creating morlet filters

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            wave_vectors -- directions of the wave part of the morlet wavelet
            gaussian_bases -- bases of the gaussian part of the morlet wavelet
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)
        Returns:
            filters -- the wavelet filters before normalization
            
    """
    n_filters, n_dim = wave_vectors.shape
    assert gaussian_bases.shape == (n_filters, n_dim, n_dim)

    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)
    else:
        shape = grid_or_shape.shape[1:]
        grid = grid_or_shape

    waves = torch.exp(1.0j * torch.matmul(grid.T, wave_vectors.T).T)
    gaussian_directions = torch.matmul(grid.T, gaussian_bases.T.reshape(n_dim, n_dim * n_filters)).T
    gaussian_directions = gaussian_directions.reshape((n_dim, n_filters) + shape)
    radii = torch.norm(gaussian_directions, dim=0)
    gaussians = torch.exp(-0.5 * radii ** 2)
    signal_dims = list(range(1, n_dim + 1))
    gabors = gaussians * waves

    if morlet:
        gaussian_sums = gaussians.sum(dim=signal_dims, keepdim=True)
        gabor_sums = gabors.sum(dim=signal_dims, keepdim=True).real
        morlets = gabors - gabor_sums / gaussian_sums * gaussians
        filters = morlets
    else:
        filters = gabors
    if ifftshift:
        filters = torch.fft.ifftshift(filters, dim=signal_dims)
        gabors = torch.fft.ifftshift(gabors, dim=signal_dims)	
    if fft:
        filters = torch.fft.fftn(filters, dim=signal_dims)
        gabors = torch.fft.fftn(gabors, dim=signal_dims)	
    if return_gabors:
        return filters, gabors
    else:
        return filters


###############################################################################
# Implementation of parametric scattering - Updating dictionaires
# -----------------------------------------------------------------------------
# Kymatio utilizes dictionaries to store wavelets. These dictionaries are fed into 
# the core scattering algorithm, where the wavelets and meta information are used 
# to compute scattering coefficents. 
#
# We will need to update the dictionaries prior 
# to the start of training so that gradients can flow from wavelets to wavelet 
# parameters, as currently the wavelets that live in the dictionaries are numpy 
# arrays. Note also that every iteration of training we will update these 
# dictionaries. 
#
# We define a series of helper functions for this purpose.


def create_filters_params(J, L):
    """ Create reusable tight frame initialized filter parameters: orientations, xis, sigmas, sigmas     

        Parameters:
            J -- scale of the scattering
            L -- number of orientation for the scattering
        Returns:
            params -- list that contains the parameters of the filters

    """
    orientations = []
    xis = []
    sigmas = []
    slants = []

    for j in range(J):
        for theta in range(L):
            sigmas.append(0.8 * 2**j)
            t = ((int(L-L/2-1)-theta) * np.pi / L)
            xis.append(3.0 / 4.0 * np.pi /2**j)
            slant = 4.0/L
            slants.append(slant)
            orientations.append(t) 

    xis = torch.tensor(xis, requires_grad=True, dtype=torch.float32)
    sigmas = torch.tensor(sigmas, requires_grad=True, dtype=torch.float32)
    slants = torch.tensor(slants, requires_grad=True, dtype=torch.float32)
    orientations = torch.tensor(orientations, requires_grad=True, dtype=torch.float32)  
    params = [orientations, xis, sigmas, slants]

    return  params


def periodize_filter_fft(x, res):
    """ Periodize the filter in fourier space

        Parameters:
            x -- signal to periodize in Fourier 
            res -- resolution to which the signal is cropped
        Returns:
            periodized -- It returns a crop version of the filter, assuming that the convolutions
                          will be done via compactly supported signals.           
    """

    s1, s2 = x.shape[0], x.shape[1]
    periodized = x.reshape(res*2, s1// 2**res, res*2, s2//2**res).mean(dim=(0,2))
    return periodized 


def update_psi(J, psi, wavelets):
    """ Update the psi dictionnary with the new wavelets

        Parameters:
            J -- scale for the scattering
            psi -- dictionnary of filters
            wavelets -- wavelet filters
        Returns:
            psi -- dictionnary of filters
    """
    wavelets = wavelets.real.contiguous().unsqueeze(3)
   
    if J == 2:
        for i,d in enumerate(psi):
                d[0] = wavelets[i]
    else:
        for i,d in enumerate(psi):
            for res in range(0, J-1):
                if res in d.keys():
                    if res == 0:
                        d[res] = wavelets[i]
                    else:
                        d[res] = periodize_filter_fft(wavelets[i].squeeze(2), res).unsqueeze(2)
    return psi


def update_wavelets_psi(J, L, psi, shape, params_filters):
        """ Create wavelets and update the psi dictionnary with the new wavelets

            Parameters:
                J -- scale for the scattering
                psi -- dictionnary of filters
                shape -- shape of the scattering (scattering.M_padded, scattering.N_padded,)
                params_filters -- the parameters used to create wavelets

            Returns:
                psi -- dictionnary of filters
                wavelets -- wavelets filters
        """
        wavelets  = morlets(shape, params_filters[0], params_filters[1],
                                   params_filters[2], params_filters[3])
        psi = update_psi(J, psi, wavelets)
        return psi, wavelets


###############################################################################
# Implementation of parametric scattering - Parametric scattering module
# -----------------------------------------------------------------------------
# We have four considerations to take into account 
#
# 1. We already have code that we can re-use; i.e. the core scattering algorithm, 
# inheritance of nn.Module. Thus we inherit from Scattering2DTorch.
#
# 2. With a parametric scattering, we want to optimize our wavelet parameters,
# so we want gradients to flow from wavelets, but we dont want to optimize 
# our wavelets. To solve this, we register the wavelet parameters as a nn.Parameter 
# wrapped together in a nn.ParameterList, and register the wavelets as module buffers.
#
# 3. At the end of every training iteration, the wavelets in our dictionaries will 
# be stale. We want to use our updated wavelet params to regenerate the wavelets 
# every training iteration. We utilize a forward_hook piror to every forward pass.
#
# 4. We want to reshape our representation to be fed into a 2D Batchnorm.
# For convinence, we hook at the end of each forward pass to reshape the output.

def _register_single_filter(self, v, n):
    self.register_buffer('tensor' + str(n), v)

class ParametricScattering2D(Scattering2D):
    """
    A learnable scattering nn.module 
    """

    def __init__(self, J, N, M, L=8):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
        """
        super(ParametricScattering2D, self).__init__(J=J, shape=(M, N), L=L)

        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)
        self.scatteringTrain = True

        self.n_coefficients =  L*L*J*(J-1)//2 + 1 + L*J  
        
        self.params_filters = create_filters_params(J, L) #kymatio init

        shape = (self.M_padded, self.N_padded,)
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)

        self.psi, _ = update_wavelets_psi(J, L, self.psi, shape, self.params_filters)
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] = torch.from_numpy(phi).unsqueeze(-1)

        self.register_single_filter = types.MethodType(_register_single_filter, self)
        self.register_filters()
        for i in range(0, len(self.params_filters)):
            self.params_filters[i] = nn.Parameter(self.params_filters[i])
            self.register_parameter(name='scattering_params_'+str(i), param=self.params_filters[i])
        self.params_filters = nn.ParameterList(self.params_filters)
        self.register_buffer(name='grid', tensor=grid)
        
        def updateFilters_hook(self, ip):
            """Update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (self.training or self.scatteringTrain):
                _, psi = self.load_filters()
                wavelets = morlets(self.grid, 
                            self.scattering_params_0,
                            self.scattering_params_1,
                            self.scattering_params_2, 
                            self.scattering_params_3)

                self.psi = update_psi(self.J, psi, wavelets)
                self.register_filters()
                self.scatteringTrain = self.training
        self.pre_hook = self.register_forward_pre_hook(updateFilters_hook)

        def reshape_hook(self, x, S):
            S = S[:,:, -self.n_coefficients:,:,:]
            S = S.reshape(S.size(0), -1, S.size(3), S.size(4))
            return S
        self.register_forward_hook(reshape_hook)


###############################################################################
# We visualize our wavelets prior to training and after. We look at 
# the Littlewood Paley diagram. Dekha means "Look! See!" in Punjabi. 

def littlewood_paley_dekha(S, display=True):
    """Plots each wavelet in Fourier space, creating a 
    comprehensive view of the scattering filterbank."""
    wavelets = morlets(S.grid, S.params_filters[0], 
                                      S.params_filters[1], S.params_filters[2], 
                                        S.params_filters[3]).detach().numpy()
    gabors = morlets(S.grid, S.params_filters[0], 
                                      S.params_filters[1], S.params_filters[2], 
                                      S.params_filters[3], return_gabors=True).cpu().detach().numpy() 
    grid = S.grid
    lp = (np.abs(wavelets) ** 2).sum(0)
    fig, ax = plt.subplots()
    
    plt.imshow(np.fft.fftshift(lp))
    if display:
        grid = grid + 18
        for i in np.abs(gabors) ** 2:
            wave = np.fft.fftshift(i)
            ax.contour(grid[1], grid[0], wave, 1, colors='white')
    plt.show()


###############################################################################
# Lets initailize our newly implemented parametric scattering and look at the 
# Littlewood-Paley diagram.
LS = ParametricScattering2D(J=2, M=28, N=28, L=8)
littlewood_paley_dekha(LS)


###############################################################################
# Training the Parametric Scattering
# -----------------------------------------------------------------------------
# We train the parametric scattering on MNSIT digits and then evaluate. 

def train(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
    
        optimizer.step()
    
        pred = output.max(1, keepdim = True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        running_loss += loss.cpu().detach()
    print("Running loss: {:.4f}, Accuracy: {:.2f}".format(running_loss, 100 * correct/len(train_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    print("Accuracy on test set: {:.2f}".format(100. * correct / len(test_loader.dataset)))

batch_size = 1024
epochs = 20

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = LS.n_coefficients
coef = int(LS.M_coefficient)
a = nn.Linear(K*coef*coef, 10)
LS_network = nn.Sequential(LS, nn.BatchNorm2d(K), nn.Flatten(), nn.Linear(K*coef*coef, 10)).to(device)
optimizer_ls = optim.SGD(LS_network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
test(LS_network, device, test_loader)

for epoch in range(0, epochs):
    train(LS_network, device, train_loader, optimizer_ls)

test(LS_network, device, test_loader)

littlewood_paley_dekha(LS)
