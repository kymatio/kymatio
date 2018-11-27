"""
Classification of spoken digit recordings
=========================================

In this example we use the 1D scattering transform to represent spoken digits,
which we then classify using a simple classifier. This shows that 1D scattering
representations are useful for this type of problem.

This dataset is automatically downloaded and preprocessed from
https://github.com/Jakobovski/free-spoken-digit-dataset.git

Downloading and precomputing scattering coefficients should take about 5 min.
Running the gradient descent takes about 1 min.

Results:
Training accuracy = 99.7%
Testing accuracy = 98.0%
"""

###############################################################################
# Preliminaries
# -------------
#
# Since kymatio handles PyTorch arrays, we first import `torch`.

import torch

###############################################################################
# We will be constructing a logistic regression classifier on top of the
# scattering coefficients, so we need some of the neural network tools from
# `torch.nn` and the Adam optimizer from `torch.optim`.

from torch.nn import Linear, NLLLoss, LogSoftmax, Sequential
from torch.optim import Adam

###############################################################################
# To handle audio file I/O, we import `os` and `scipy.io.wavfile`. We also need
# `numpy` for some basic array manipulation.

from scipy.io import wavfile
import os
import numpy as np

###############################################################################
# To evaluate our results, we need to form a confusion matrix using
# scikit-learn and display them using `matplotlib`.

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

###############################################################################
# Finally, we import the `Scattering1D` class from the `scattering` package and
# the `fetch_fsdd` function from `scattering.datasets`. The `Scattering1D`
# class is what lets us calculate the scattering transform, while the
# `fetch_fsdd` function downloads the FSDD, if needed.

from kymatio import Scattering1D
from kymatio.datasets import fetch_fsdd

###############################################################################
# Pipeline setup
# --------------
# We start by specifying the dimensions of our processing pipeline along with
# some other parameters.
#
# First, we have signal length. Longer signals are truncated and shorter
# signals are zero-padded. The sampling rate is 8000 Hz, so this corresponds to
# little over a second.
T = 2**13

###############################################################################
# Maximum scale 2**J of the scattering transform (here, about 30 milliseconds)
# and the number of wavelets per octave.
J = 8
Q = 12

###############################################################################
# We need a small constant to add to the scattering coefficients before
# computing the logarithm. This prevents very large values when the scattering 
# coefficients are very close to zero.
log_eps = 1e-6

###############################################################################
# If a GPU is available, let's use it!
use_cuda = torch.cuda.is_available()

###############################################################################
# For reproducibility, we fix the seed of the random number generator.
torch.manual_seed(42)

###############################################################################
# Loading the data
# ----------------
# Once the parameter are set, we can start loading the data into a format that
# can be fed into the scattering transform and then a logistic regression
# classifier.
#
# We first download the dataset. If it's already downloaded, `fetch_fsdd` will
# simply return the information corresponding to the dataset that's already
# on disk.
info_data = fetch_fsdd()
files = info_data['files']
path_dataset = info_data['path_dataset']

###############################################################################
# Set up Tensors to hold the audio signals (`x_all`), the labels (`y_all`), and
# whether the signal is in the train or test set (`subset`).
x_all = torch.zeros(len(files), 1, T, dtype=torch.float32)
y_all = torch.zeros(len(files), dtype=torch.int64)
subset = torch.zeros(len(files), dtype=torch.int64)

###############################################################################
# For each file in the dataset, we extract its label `y` and its index from the
# filename. If the index is between 0 and 4, it is placed in the test set, while
# files with larger indices are used for training. The actual signals are
# normalized to have maximum amplitude one, and are truncated or zero-padded
# to the desired length `T`. They are then stored in the `x_all` Tensor while
# their labels are in `y_all`.
for k, f in enumerate(files):
    basename = f.split('.')[0]

    # Get label (0-9) of recording.
    y = int(basename.split('_')[0])

    # Index larger than 5 gets assigned to training set.
    if int(basename.split('_')[2]) >= 5:
        subset[k] = 0
    else:
        subset[k] = 1

    # Load the audio signal and normalize it.
    _, x = wavfile.read(os.path.join(path_dataset, f))
    x = np.asarray(x, dtype='float')
    x /= np.max(np.abs(x))

    # Convert from NumPy array to PyTorch Tensor.
    x = torch.from_numpy(x)

    # If it's too long, truncate it.
    if x.numel() > T:
        x = x[:T]

    # If it's too short, zero-pad it.
    start = (T - x.numel()) // 2

    x_all[k,0,start:start + x.numel()] = x
    y_all[k] = y

###############################################################################
# Log-scattering transform
# ------------------------
# We now create the `Scattering1D` object that will be used to calculate the
# scattering coefficients.
scattering = Scattering1D(J, T, Q)

###############################################################################
# If we are using CUDA, the scattering transform object must be transferred to
# the GPU by calling its `cuda()` method. The data is similarly transferred.
if use_cuda:
    scattering.cuda()
    x_all = x_all.cuda()
    y_all = y_all.cuda()

###############################################################################
# Compute the scattering transform for all signals in the dataset.
Sx_all = scattering.forward(x_all)

###############################################################################
# Since it does not carry useful information, we remove the zeroth-order
# scattering coefficients, which are always placed in the first channel of
# the scattering Tensor.
Sx_all = Sx_all[:,1:,:]

###############################################################################
# To increase discriminability, we take the logarithm of the scattering
# coefficients (after adding a small constant to make sure nothing blows up
# when scattering coefficients are close to zero).
Sx_all = torch.log(torch.abs(Sx_all) + log_eps)

###############################################################################
# Finally, we average along the last dimension (time) to get a time-shift
# invariant representation.
Sx_all = torch.mean(Sx_all, dim=-1)

###############################################################################
# Training the classifier
# -----------------------
# With the log-scattering coefficients in hand, we are ready to train our
# logistic regression classifier.
#
# First, we extract the training data (those for which `subset` equals `0`)
# and the associated labels.
Sx_tr, y_tr = Sx_all[subset == 0], y_all[subset == 0]

###############################################################################
# Standardize the data to have mean zero and unit variance. Note that we need
# to apply the same transformation to the test data later, so we save the
# mean and standard deviation Tensors.
mu_tr = Sx_tr.mean(dim=0)
std_tr = Sx_tr.std(dim=0)
Sx_tr = (Sx_tr - mu_tr) / std_tr

###############################################################################
# Here we define a Logistic Regression model using PyTorch. We train it using
# Adam with a negative log-likelihood loss.
num_input = Sx_tr.shape[-1]
num_classes = y_tr.cpu().unique().numel()
model = Sequential(Linear(num_input, num_classes), LogSoftmax(dim=1))
optimizer = Adam(model.parameters())
criterion = NLLLoss()

###############################################################################
# As before, if we're on a GPU, transfer the model and the loss function onto
# the device.
if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

###############################################################################
# Before training the model, we set some parameters for the optimization
# procedure.

# Number of signals to use in each gradient descent step (batch).
batch_size = 32
# Number of epochs.
num_epochs = 50
# Learning rate for Adam.
lr = 1e-4

###############################################################################
# Given these parameters, we compute the total number of batches.
nsamples = Sx_tr.shape[0]
nbatches = nsamples // batch_size

###############################################################################
# Now we're ready to train the classifier.
for e in range(num_epochs):
    # Randomly permute the data. If necessary, transfer the permutation to the
    # GPU.
    perm = torch.randperm(nsamples)
    if use_cuda:
        perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        model.zero_grad()
        resp = model.forward(Sx_tr[idx])
        loss = criterion(resp, y_tr[idx])
        loss.backward()
        optimizer.step()

    # Calculate the response of the training data at the end of this epoch and
    # the average loss.
    resp = model.forward(Sx_tr)
    avg_loss = criterion(resp, y_tr)

    # Try predicting the classes of the signals in the training set and compute
    # the accuracy.
    y_hat = resp.argmax(dim=1)
    accuracy = (y_tr == y_hat).float().mean()

    print('Epoch {}, average loss = {:1.3f}, accuracy = {:1.3f}'.format(
        e, avg_loss, accuracy))

###############################################################################
# Now that our network is trained, let's test it!
#
# First, we extract the test data (those for which `subset` equals `1`) and the
# associated labels.
Sx_te, y_te = Sx_all[subset == 1], y_all[subset == 1]

###############################################################################
# Use the mean and standard deviation calculated on the training data to 
# standardize the testing data, as well.
Sx_te = (Sx_te - mu_tr) / std_tr

###############################################################################
# Calculate the response of the classifier on the test data and the resulting
# loss.
resp = model.forward(Sx_te)
avg_loss = criterion(resp, y_te)

# Try predicting the labels of the signals in the test data and compute the
# accuracy.
y_hat = resp.argmax(dim=1)
accu = (y_te == y_hat).float().mean()

print('TEST, average loss = {:1.3f}, accuracy = {:1.3f}'.format(
      avg_loss, accu))

###############################################################################
# Plotting the classification accuracy as a confusion matrix
# ----------------------------------------------------------
# Let's see what the very few misclassified sounds get misclassified as. We
# will plot a confusion matrix which indicates in a 2D histogram how often
# one sample was mistaken for another (anything on the diagonal is correctly
# classified, anything off the diagonal is wrong).

predicted_categories = y_hat.cpu().numpy()
actual_categories = y_te.cpu().numpy()

confusion = confusion_matrix(actual_categories, predicted_categories)
plt.figure()
plt.imshow(confusion)
tick_locs = np.arange(10)
ticks = ['{}'.format(i) for i in range(1, 11)]
plt.xticks(tick_locs, ticks)
plt.yticks(tick_locs, ticks)
plt.ylabel("True number")
plt.xlabel("Predicted number")
plt.show()
