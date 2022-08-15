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
# Since we're using TensorFlow and Keras to train the model, import the
# relevant modules.

import tensorflow as tf

from tensorflow.keras import layers

###############################################################################
# To handle audio file I/O, we import `os` and `scipy.io.wavfile`. We also need
# `numpy` for some basic array manipulation.

from scipy.io import wavfile
import os
import numpy as np

###############################################################################
# Finally, we import the `Scattering1D` class from the `kymatio.keras` package
# and the `fetch_fsdd` function from `kymatio.datasets`. The `Scattering1D`
# class is what lets us calculate the scattering transform, while the
# `fetch_fsdd` function downloads the FSDD, if needed.

from kymatio.keras import Scattering1D
from ...datasets import fetch_fsdd

###############################################################################
# Pipeline setup
# --------------
# We start by specifying the dimensions of our processing pipeline along with
# some other parameters.
#
# First, we have signal length. Longer signals are truncated and shorter
# signals are zero-padded. The sampling rate is 8000 Hz, so this corresponds to
# little over a second.

T = 2 ** 13

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
# Set up NumPy arrays to hold the audio signals (`x_all`), the labels
# (`y_all`), and whether the signal is in the train or test set (`subset`).

x_all = np.zeros((len(files), T))
y_all = np.zeros(len(files), dtype=np.uint8)
subset = np.zeros(len(files), dtype=np.uint8)

###############################################################################
# For each file in the dataset, we extract its label `y` and its index from the
# filename. If the index is between 0 and 4, it is placed in the test set, while
# files with larger indices are used for training. The actual signals are
# normalized to have maximum amplitude one, and are truncated or zero-padded
# to the desired length `T`. They are then stored in the `x_all` array while
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

    # If it's too long, truncate it.
    if len(x) > T:
        x = x[:T]

    # If it's too short, zero-pad it.
    start = (T - len(x)) // 2

    x_all[k,start:start + len(x)] = x
    y_all[k] = y

###############################################################################
# Log-scattering layer
# --------------------
# We now create a classification model using the `Scattering1D` Keras layer.
# First, we take the input signals of length `T`.

x_in = layers.Input(shape=(T))

###############################################################################
# These are fed into the `Scattering1D` layer.

x = Scattering1D(J, Q=Q)(x_in)

###############################################################################
# Since it does not carry useful information, we remove the zeroth-order
# scattering coefficients, which are always placed in the first channel of
# the scattering transform.

x = layers.Lambda(lambda x: x[..., 1:, :])(x)

# To increase discriminability, we take the logarithm of the scattering
# coefficients (after adding a small constant to make sure nothing blows up
# when scattering coefficients are close to zero). This is known as the
# log-scattering transform.

x = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))(x)

###############################################################################
# We then average along the last dimension (time) to get a time-shift
# invariant representation.

x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)

###############################################################################
# Finally, we apply batch normalization to ensure that the data is within a
# moderate range.

x = layers.BatchNormalization(axis=1)(x)

###############################################################################
# These features are then used to classify the input signal using a dense
# layer followed by a softmax activation.

x_out = layers.Dense(10, activation='softmax')(x)

###############################################################################
# Finally, we create the model and display it.

model = tf.keras.models.Model(x_in, x_out)
model.summary()

###############################################################################
# Training the classifier
# -----------------------
# Having set up the model, we attach an Adam optimizer and a cross-entropy
# loss function.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################################################
# We then train the model using `model.fit`. The training data is given by
# those indices satisfying `subset == 0`.

model.fit(x_all[subset == 0], y_all[subset == 0], epochs=50,
          batch_size=64, validation_split=0.2)

###############################################################################
# Finally, we evaluate the model on the held-out test data. These are given by
# the indices `subset == 1`.

model.evaluate(x_all[subset == 1], y_all[subset == 1], verbose=2)
