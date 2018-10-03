import numpy as np
import time
import torch
import os

from sklearn import linear_model, model_selection
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import pdist
from scattering.scattering import Scattering3D
from scattering.scattering3d.utils import (compute_integrals, 
                                           generate_weighted_sum_of_gaussians)
from scattering.datasets import fetch_qm7
from scattering.caching import get_cache_dir


def evaluate_linear_regression(scattering_coef, target):
    """
    Evaluates linear ridge regression on molecule properties given scattering coefficients
    for different values of the l_2 penalty and prints the results.

    Parameters
    ----------
    scattering_coef:  numpy array
        N x D
        N: number of samples
        D: number of features (scattering coefficients)
    target: numpy array
        N x 1
        target property

    """
    alphas = 10.**(-np.arange(5, 15))
    maes = np.zeros_like(alphas)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
                                        scattering_coef, target, test_size=0.2)
    for i, alpha in enumerate(alphas):
        ridge = linear_model.Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)
        maes[i] = mean_absolute_error(y_test, ridge.predict(x_test))
    print('Ridge regression :')
    for i in range(len(alphas)):
        print(alphas[i], maes[i])
    return maes

def get_valence(charges):
    """
    Returns the number valence electrons of a particle given the nuclear charge
    Parameters
    ----------
    charges: integer numpy array

    Returns
    -------
    output : integer numpy array of the same size as charges
    """
    return (
        charges * (charges <= 2) +
        (charges - 2) * np.logical_and(charges > 2, charges <= 10) +
        (charges - 10) * np.logical_and(charges > 10, charges <= 18))


def renormalize(positions, charges, sigma, overlapping_precision=1e-1):
    """
    Normalizes distances of particles in a density so that they can be 
    positioned in an integer grid.

    Parameters
    ----------
    positions : float array
        input 3D numpy array of
        size N x A x 3
        where N is the number of data points, A is the number of particles,
        and 3 are the x,y,z coordinates of each particle
    charges : integer array
        size N x A
        the charge of each particle
    sigma : float
        controls the size/std of the gaussian that represents a particle
    overlapping_precision: float
        sets an upper bound on the overlap of the two nearest gaussians in the 
        resulting densities

    Returns
    -------
    output : float array
        array of the same size and type as positions normalized to fit in an 
        integer grid
    """
    min_dist = np.inf

    # we loop over the data to identify the smallest distance between any two 
    # particles
    for i in range(positions.shape[0]):
        n_atoms = np.sum(charges[i] != 0)
        pos = positions[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    print(delta / min_dist)

    return positions * delta / min_dist


def get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma):
    qm7 = fetch_qm7(align=True)
    positions = qm7['positions']
    charges = qm7['charges'].astype('float32')
    energies = qm7['energies']
    valence_charges = get_valence(charges)

    positions = renormalize(positions, charges, sigma)

    return (torch.from_numpy(positions), torch.from_numpy(energies),
            torch.from_numpy(charges), torch.from_numpy(valence_charges))


# """Trains a simple linear regression model with solid harmonic
# scattering coefficients on the atomisation energies of the QM7
# database.

# Achieves a MAE of ... kcal.mol-1
# """
cuda = torch.cuda.is_available()
batch_size = 32
M, N, O = 192, 128, 96
grid = torch.from_numpy(
    np.fft.ifftshift(
        np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'),
        axes=(1, 2, 3)))
sigma = 1.5 # the standard deviation of the gaussian representing a particle
J, L = 1, 2 # the number of scales and the number of different angular 
            # oscillations of the solid harmonic wavelets
integral_powers = [0.5,1.,2.] # the number of different integral powers used 
                              # as features
(pos, energies, 
full_charges, valence_charges) = get_qm7_positions_energies_and_charges(
                                                         M, N, O, J, L, sigma)
if cuda:
    grid = grid.cuda()
    pos = pos.cuda()
    energies = energies.cuda()
    full_charges = full_charges.cuda()
    valence_charges = valence_charges.cuda()
n_molecules = pos.size(0)
n_batches = np.ceil(n_molecules / batch_size).astype(int)

scattering = Scattering3D(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

order_0, order_1, order_2 = [], [], []
print('Computing solid harmonic scattering coefficients of molecules'
        'of QM7 database on {}'.format('GPU' if cuda else 'CPU'))
print('L: {}, J: {}, integral powers: {}'.format(L, J, integral_powers))

this_time = None
last_time = None
for i in range(n_batches):
    this_time = time.time()
    if last_time is not None:
        dt = this_time - last_time
        completion_factor = float(i) / n_batches
        print("Iteration {} ETA: [{:02}:{:02}:{:02}]".format(
                      i + 1, int(((n_batches - i - 1) * dt) // 3600),
                      int((((n_batches - i - 1) * dt) // 60) % 60),
                      int(((n_batches - i - 1) * dt) % 60)), end='\r')
    else:
        print("Iteration {} ETA: {}".format(i + 1,'-'),end='\r')
    last_time = this_time
    time.sleep(1)

    start, end = i * batch_size, min((i + 1) * batch_size, n_molecules)

    pos_batch = pos[start:end]
    full_batch = full_charges[start:end]

    full_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, full_batch, sigma, cuda=cuda)

    full_order_0 = compute_integrals(full_density_batch, integral_powers)
    full_order_1, full_order_2 = scattering(full_density_batch, order_2=True,
                            method='integral', integral_powers=integral_powers)

    full_order_0 = compute_integrals(full_density_batch, integral_powers)
    val_batch = valence_charges[start:end]
    val_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, val_batch, sigma, cuda=cuda)
    val_order_0 = compute_integrals(val_density_batch, integral_powers)
    val_order_1, val_order_2 = scattering(val_density_batch, order_2=True,
                            method='integral', integral_powers=integral_powers)

    core_density_batch = full_density_batch - val_density_batch
    core_order_0 = compute_integrals(core_density_batch, integral_powers)
    core_order_1, core_order_2 = scattering(core_density_batch, order_2=True,
                            method='integral', integral_powers=integral_powers)

    order_0.append(
        torch.stack([full_order_0, val_order_0, core_order_0], dim=-1))
    order_1.append(
        torch.stack([full_order_1, val_order_1, core_order_1], dim=-1))
    order_2.append(
        torch.stack([full_order_2, val_order_2, core_order_2], dim=-1))

order_0 = torch.cat(order_0, dim=0)
order_1 = torch.cat(order_1, dim=0)
order_2 = torch.cat(order_2, dim=0)

np_order_0 = order_0.numpy().reshape((n_molecules, -1))
np_order_1 = order_1.numpy().reshape((n_molecules, -1))
np_order_2 = order_2.numpy().reshape((n_molecules, -1))

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)
cachedir = get_cache_dir("qm7/experiments")
np.save(os.path.join(cachedir, 'order_0_' + basename), np_order_0)
np.save(os.path.join(cachedir, 'order_1_' + basename), np_order_1)
np.save(os.path.join(cachedir, 'order_2_' + basename), np_order_2)

scattering_coef = np.concatenate([np_order_0, np_order_1, np_order_2], axis=1)
if cuda:
    energies = energies.cpu()
target = energies.numpy()

print('order 1 : {} coef, order 2 : {} coefs'.format(np_order_1.shape[1],
                                                     np_order_2.shape[1]))

maes = evaluate_linear_regression(scattering_coef, target)



