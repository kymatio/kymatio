"""
Classification of MNIST with scattering
=======================================
Here we demonstrate a simple application of scattering on the MNIST dataset.
We use 10000 images to train a linear classifier. Features are normalized by
batch normalization.
"""

###############################################################################
# Preliminaries
# -------------
#
# Since we're using TensorFlow and Keras to train the model, import the
# relevant modules.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

###############################################################################
# Finally, we import the `Scattering2D` class from the `kymatio.keras`
# package.

from kymatio.keras import Scattering2D

###############################################################################
# Training and testing the model
# ------------------------------
#
# First, we load in the data and normalize it.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

###############################################################################
# We then create a Keras model using the scattering transform followed by a
# dense layer and a softmax activation.

inputs = Input(shape=(28, 28))
x = Scattering2D(J=3, L=8)(inputs)
x = Flatten()(x)
x_out = Dense(10, activation='softmax')(x)
model = Model(inputs, x_out)

###############################################################################
# Display the created model.

model.summary()

###############################################################################
# Once the model is created, we couple it with an Adam optimizer and a
# cross-entropy loss function.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################################################
# We then train the model using `model.fit` on a subset of the MNIST data.

model.fit(x_train[:10000], y_train[:10000], epochs=15,
          batch_size=64, validation_split=0.2)

###############################################################################
# Finally, we evaluate the model on the held-out test data.

model.evaluate(x_test, y_test)
