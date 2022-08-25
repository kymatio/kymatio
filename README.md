Kymatio: Wavelet scattering in Python - v0.3.0 “Erdre”
======================================================

Kymatio is an implementation of the wavelet scattering transform in the Python programming language, suitable for large-scale numerical experiments in signal processing and machine learning.
Scattering transforms are translation-invariant signal representations implemented as convolutional networks whose filters are not learned, but fixed (as wavelet filters).

[![PyPI](https://img.shields.io/badge/Python-3.7%2C_3.8%2C_3.9%2C_3.10-blue.svg)](https://pypi.org/project/kymatio/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/kymatio/kymatio.svg?branch=master)](https://travis-ci.org/kymatio/kymatio)
[![Downloads](https://pepy.tech/badge/kymatio)](https://pepy.tech/project/kymatio)
[![codecov](https://codecov.io/gh/kymatio/kymatio/branch/master/graph/badge.svg)](https://codecov.io/gh/kymatio/kymatio)


Use Kymatio if you need a library that:
* supports 1-D, 2-D, and 3-D wavelets,
* integrates wavelet scattering in a deep learning architecture, and
* runs seamlessly on CPU and GPU hardware, with major deep learning APIs, such
  as PyTorch and TensorFlow.

# The Kymatio environment

## Flexibility

The Kymatio organization associates the developers of several pre-existing packages for wavelet scattering, including `ScatNet`, `scattering.m`, `PyScatWave`, `WaveletScattering.jl`, and `PyScatHarm`.

The resort to PyTorch tensors as inputs to Kymatio allows the programmer to backpropagate the gradient of wavelet scattering coefficients, thus integrating them within an end-to-end trainable pipeline, such as a deep neural network.

## Portability

Each of these algorithms is written in a high-level imperative paradigm, making it portable to any Python library for array operations as long as it enables complex-valued linear algebra and a fast Fourier transform (FFT).

Each algorithm comes packaged with a frontend and backend. The frontend takes care of
interfacing with the user. The backend defines functions necessary for
computation of the scattering transform.

Currently, there are eight available frontend–backend pairs, NumPy (CPU), scikit-learn (CPU), pure PyTorch (CPU and GPU), PyTorch>=1.10 (CPU and GPU), PyTorch+scikit-cuda (GPU), PyTorch>=1.10+scikit-cuda (GPU), TensorFlow (CPU and GPU), Keras (CPU and GPU), and Jax (CPU and GPU).

## Scalability

Kymatio integrates the construction of wavelet filter banks in 1D, 2D, and 3D, as well as memory-efficient algorithms for extracting wavelet scattering coefficients, under a common application programming interface.

Running Kymatio on a graphics processing unit (GPU) rather than a multi-core conventional central processing unit (CPU) allows for significant speedups in computing the scattering transform.
The current speedup with respect to CPU-based MATLAB code is of the order of 10 in 1D and 3D and of the order of 100 in 2D.

We refer to our [official benchmarks](https://www.kymat.io/userguide.html#benchmarks) for further details.

## How to cite

If you use this package, please cite our paper <a href="">Kymatio: Scattering Transforms in Python</a>:

Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Chaudhary M., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2020). Kymatio: Scattering Transforms in Python. Journal of Machine Learning Research 21(60):1−6, 2020. [(paper)](http://jmlr.org/papers/v21/19-047.html) [(bibtex)](http://jmlr.org/papers/v21/19-047.bib)


# Installation


## Dependencies

Kymatio requires:

* Python (>= 3.7)
* SciPy (>= 0.13)


### Standard installation
We strongly recommend running Kymatio in an Anaconda environment, because this simplifies the installation of other
dependencies. You may install the latest version of Kymatio using the package manager `pip`, which will automatically download
Kymatio from the Python Package Index (PyPI):

```
pip install kymatio
```

Linux and macOS are the two officially supported operating systems.


# Frontend

## NumPy

To explicitly call the NumPy frontend, run:

```
from kymatio.numpy import Scattering2D
scattering = Scattering2D(J=2, shape=(32, 32))
```

## Scikit-learn

After installing the latest version of scikit-learn, you can call `Scattering2D` as a `Transformer` using:

```
from kymatio.sklearn import Scattering2D

scattering_transformer = Scattering2D(2, (32, 32))
```

## PyTorch

After installing the latest version of PyTorch, you can call `Scattering2D` as a `torch.nn.Module` using:

```
from kymatio.torch import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32))
```

## TensorFlow

After installing the latest version of TensorFlow, you can call `Scattering2D` as a `tf.Module` using:

```
from kymatio.tensorflow import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32))
```

## Keras

Alternatively, with TensorFlow installed, you can call `Scattering2D` as a Keras `Layer` using:

```
from tensorflow.keras.layers import Input
from kymatio.keras import Scattering2D

inputs = Input(shape=(32, 32))
scattering = Scattering2D(J=2)(inputs)
```

## Jax

With Jax installed, you can also instantiate a Jax `Scattering2D` object:

```
from kymatio.jax import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32))
```

# Installation from source

Assuming the Kymatio source has been downloaded, you may install it by running

```
pip install -r requirements.txt
python setup.py install
```

Developers can also install Kymatio via:

```
pip install -r requirements.txt
python setup.py develop
```


## GPU acceleration

Certain frontends, `numpy` and `sklearn`, only allow processing on the CPU and are therefore slower. The `torch`, `tensorflow`, `keras`, and `jax` frontends, however, also support GPU processing, which can significantly accelerate computations. Additionally, the `torch` backend supports an optimized `skcuda` backend which currently provides the fastest performance in computing scattering transforms. In 2D, it may be instantiated using:

```
from kymatio.torch import Scattering2D

scattering = Scattering2D(J=2, shape=(32, 32), backend='torch_skcuda')
```

This is particularly useful when working with large images, such as those in ImageNet, which are of size 224×224.

## PyTorch and scikit-cuda

To run Kymatio on a graphics processing unit (GPU), you can either use the PyTorch-style `cuda()` method to move your
object to GPU. Kymatio is designed to operate on a variety of backends for tensor operations. For extra speed, install
the CUDA library and the `skcuda` dependency by running the following pip command:

```
pip install scikit-cuda cupy
```

The user may control the choice of backend at runtime via for instance:

```
from kymatio.torch import Scattering2D
scattering = Scattering2D(J=2, shape=(32, 32)), backend='torch_skcuda')
```

# Documentation

The documentation of Kymatio is officially hosted on the [kymat.io](https://www.kymat.io/) website.


## Online resources

* [GitHub repository](https://github.com/kymatio/kymatio)
* [GitHub issue tracker](https://github.com/kymatio/kymatio/issues)
* [BSD-3-Clause license](https://github.com/kymatio/kymatio/blob/master/LICENSE.md)
* [List of authors](https://github.com/kymatio/kymatio/blob/master/AUTHORS.md)
* [Code of conduct](https://github.com/kymatio/kymatio/blob/master/CODE_OF_CONDUCT.md)


## Building the documentation from source
The documentation can also be found in the `doc/` subfolder of the GitHub repository.
To build the documentation locally, please clone this repository and run

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

## Support

We wish to thank the Scientific Computing Core at the Flatiron Institute for the use of their computing resources for testing.

[![Flatiron](doc/source/_static/FL_Full_Logo_Mark_Small.png)](https://www.flatironinstitute.org/)

We would also like to thank École Normale Supérieure for their support.

[![ENS](https://www.ens.fr/sites/default/files/inline-images/logo.jpg)](https://www.ens.fr/)

## Kymatio

Kyma (*κύμα*) means *wave* in Greek. By the same token, Kymatio (*κυμάτιο*) means *wavelet*.

Note that the organization and the library are capitalized (*Kymatio*) whereas the corresponding Python module is written in lowercase (`import kymatio`).

The recommended pronunciation for Kymatio is *kim-ah-tio*. In other words, it rhymes with patio, not with ratio.
