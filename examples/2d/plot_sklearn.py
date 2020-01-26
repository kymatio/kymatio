"""
Scikit-learn transformer example
================================
Here we demonstrate a simple application of scattering as a transformer
"""


from kymatio.sklearn import Scattering2D
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create a scattering object (can be Scattering1D or Scattering3D as well)
S = Scattering2D(shape=(8, 8), J=1)

# Use the toy digits dataset (8x8 digits)
digits = datasets.load_digits()

# Split data into train and test subsets
images = digits.images
images = images.reshape((images.shape[0], -1))
x_train, x_test, y_train, y_test = train_test_split(images, digits.target, test_size=0.5, shuffle=False)

# Create the scikit-learn transformer

# Create a classifier: a support vector classifier
classifier = LogisticRegression()
estimators = [('scatter', S), ('clf', classifier)]
pipeline = Pipeline(estimators)

# We learn the digits on the first half of the digits
pipeline.fit(x_train, y_train)

# Now predict the value of the digit on the second half:
y_pred = pipeline.predict(x_test)

print(accuracy_score(y_test, y_pred))
