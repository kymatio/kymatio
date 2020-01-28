"""
Classification of spoken digit recordings
=========================================

This example shows how to combine the scattering transform with
tensorflow.keras for a 1d classification task. The example follows closely
the `plot_classif_torch.py` example which shows how to solve the same problem
using pytorch. More details can be found there.
"""

import tensorflow as tf
from tensorflow.keras import layers
from scipy.io import wavfile
import os
import numpy as np
from kymatio.keras import Scattering1D
from kymatio.datasets import fetch_fsdd

# %% Read the data and do some pre-processing

T = 2**13
info_data = fetch_fsdd()
files = info_data['files']
path_dataset = info_data['path_dataset']

x_all = np.zeros((len(files), T))
y_all = np.zeros(len(files), dtype=np.int64)
subset = np.zeros(len(files), dtype=np.int64)

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

# %% Construct the tensorflow.keras model

J = 8
Q = 12

log_eps = 1e-6

x_in = layers.Input(shape=(T))
S = Scattering1D(J, Q=Q)
Sx = S(x_in)
L = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))
Lx = L(Sx)
x = layers.GlobalAveragePooling1D(data_format='channels_first')(Lx)
x = layers.BatchNormalization(axis=1)(x)
x_out = layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(x_in, x_out)
model.summary()

# %% Train the model and test accuracy on held out data

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
y_all = y_all.astype(np.uint8)

model.fit(x_all[:1600], y_all[:1600], epochs=50,
          batch_size=64, validation_split=0.2)

model.evaluate(x_all[1600:], y_all[1600:], verbose=2)