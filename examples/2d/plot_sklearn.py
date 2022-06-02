"""
Scikit-learn transformer example
================================
Here we demonstrate a simple application of scattering as a transformer
"""

###############################################################################
# Preliminaries
# -------------
#
# Import the relevant classes and functions from sciki-learn.

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

###############################################################################
# Import the scikit-learn `Scattering2D` frontend.

from kymatio.sklearn import Scattering2D

###############################################################################
# Preparing the data
# ------------------
#
# First, we load the dataset. In this case, it's the UCI ML digits dataset
# included with scikit-learn, consisting of 8×8 images of handwritten digits
# from one to ten.

digits = datasets.load_digits()

###############################################################################
# We then extract the images, reshape them to an array of size `(n_features,
# n_samples)` needed for processing in a scikit-learn pipeline.

images = digits.images
images = images.reshape((images.shape[0], -1))

###############################################################################
# We then split the images (and their labels) into a train and a test set.

x_train, x_test, y_train, y_test = train_test_split(images, digits.target,
                                                    test_size=0.5, shuffle=False)

###############################################################################
# Training and testing the model
# ------------------------------
# Create a `Scattering2D` object, which implements a scikit-learn
# `Transformer`. We set the input shape to match that of the the images (8×8)
# and the averaging scale is set to `J = 1`, which means that the local
# invariance is `2 ** 1 = 1`.

S = Scattering2D(shape=(8, 8), J=1)

###############################################################################
# We then plug this into a scikit-learn pipeline which takes the scattering
# features, scales them, then provides them to a `LogisticRegression` classifier.

classifier = LogisticRegression(max_iter=150)
estimators = [('scatter', S), ('scaler', StandardScaler()), ('clf', classifier)]
pipeline = Pipeline(estimators)

###############################################################################
# Given the pipeline, we train it on `(x_train, y_train)` using
# `pipelien.fit`.

pipeline.fit(x_train, y_train)

###############################################################################
# Finally, we calculate the predicted labels on the test data and output the
# classification accuracy.

y_pred = pipeline.predict(x_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
