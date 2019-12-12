Kymatio: Wavelet scattering in PyTorch (and soon, TensorFlow too)
=================================================================

Kymatio is an implementation of the wavelet scattering transform in the Python programming language, suitable for large-scale numerical experiments in signal processing and machine learning.
A scattering transform is a nonlinear signal representation resembling the response of a deep convolutional network, yet requiring no prior training.
Indeed, in a scattering network, convolutional kernels are defined as wavelets (band-pass filters) instead of being learned from data.
This property makes Kymatio suitable to many classification and regression tasks, both in supervised and unsupervised settings.

[![PyPI](https://img.shields.io/badge/python-3.5%2C%203.6%2C%203.7-blue.svg)](https://pypi.org/project/kymatio/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/kymatio/kymatio.svg?branch=master)](https://travis-ci.org/kymatio/kymatio)
[![Downloads](https://pepy.tech/badge/kymatio)](https://pepy.tech/project/kymatio)
[![codecov](https://codecov.io/gh/kymatio/kymatio/branch/master/graph/badge.svg)](https://codecov.io/gh/kymatio/kymatio)


Use Kymatio if you need a library that:
* supports 1-D, 2-D, and 3-D wavelets,
* integrates wavelet scattering in a deep learning architecture, and
* runs seamlessly on CPU and GPU hardware.


### Flexibility

The Kymatio organization associates the developers of several pre-existing packages for wavelet scattering, including `ScatNet`, `scattering.m`, `PyScatWave`, `WaveletScattering.jl`, and `PyScatHarm`.

The resort to PyTorch tensors as inputs to Kymatio allows the programmer to backpropagate the gradient of wavelet scattering coefficients, thus integrating them within an end-to-end trainable pipeline, such as a deep neural network.

### Portability

Each of these algorithms is written in a high-level imperative paradigm, making it portable to any Python library for array operations as long as it enables complex-valued linear algebra and a fast Fourier transform (FFT).

As of v0.1, Kymatio fully supports two backends: PyTorch (CPU and GPU) and scikit-cuda (GPU only).

The next version of Kymatio, planned around spring 2020, will also support NumPy (CPU only) and TensorFlow (CPU and GPU) as backends.
You may already try out these new backends in the alpha release of Kymatio v0.2, whose source code is available on the following branch:
https://github.com/kymatio/kymatio/tree/0.2.X


### Scalability

Kymatio integrates the construction of wavelet filter banks in 1D, 2D, and 3D, as well as memory-efficient algorithms for extracting wavelet scattering coefficients, under a common application programming interface.

Running Kymatio on a graphics processing unit (GPU) rather than a multi-core conventional computer processing unit (CPU) allows for significant speedups in computing the scattering transform.
The current speedup with respect to CPU-based MATLAB code is of the order of 10 in 1D and 3D and of the order of 100 in 2D.

We refer to our [official benchmarks](https://www.kymat.io/userguide.html#benchmarks) for further details.

### How to cite

If you use this package, please cite the following paper:

Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2019). Kymatio: Scattering Transforms in Python. arXiv preprint arXiv:1812.11214. [(paper)](https://arxiv.org/abs/1812.11214)

## Installation


### Dependencies

Kymatio v0.1 requires the following packages:

* Python (>= 3.5)
* PyTorch (>= 0.4)
* SciPy (>= 0.13)

The next version, v0.2, will offer a pure NumPy backend (CPU only), hence making the the PyTorch dependency optional.


### Installation on CPU hardware
We strongly recommend running Kymatio in an Anaconda environment, because this simplifies the installation of [PyTorch](https://pytorch.org). This is most easily achieved by running

```
conda install pytorch torchvision -c pytorch
```

Once PyTorch is installed, you may install the latest version of Kymatio using the package manager `pip`, which will automatically download Kymatio from the Python Package Index (PyPI):

```
pip install kymatio
```

Linux and macOS are the two officially supported operating systems.


### Installation on GPU hardware

To run Kymatio on a graphics processing unit (GPU), you can either use the PyTorch-style `cuda()` method to move your object to GPU. For extra speed, install the CUDA library and install the `scikit-cuda` dependency by running the following pip command:

```
pip install scikit-cuda cupy
```

Then, set the `KYMATIO_BACKEND` to `skcuda`:

```
os.environ["KYMATIO_BACKEND"] = "skcuda"
```


#### Available backends: PyTorch and scikit-cuda

Kymatio is designed to operate on a variety of backends for tensor operations.
The user may control the choice of backend at runtime by setting the environment variable `KYMATIO_BACKEND`, or by editing the Kymatio configuration file (`~/.config/kymatio/kymatio.cfg` on Linux).

The two available backends are PyTorch (`torch`) and scikit-cuda (`skcuda`).

PyTorch is the default backend in 1D, 2D, and 3D scattering. However, for applications of the 2D scattering transform to large images (e.g. ImageNet, of size 224×224), we recommend the `skcuda` backend, which is substantially faster than PyTorch.

### Installation from source

Assuming PyTorch is already installed (see above) and the Kymatio source has been downloaded, you may install it by running

```
pip install -r requirements.txt
python setup.py install
```


## Documentation

The documentation of Kymatio is officially hosted on the [kymat.io](https://www.kymat.io/) website.

### Online resources

* [GitHub repository](https://github.com/kymatio/kymatio)
* [GitHub issue tracker](https://github.com/kymatio/kymatio/issues)
* [BSD-3-Clause license](https://github.com/kymatio/kymatio/blob/master/LICENSE.md)
* [Code of conduct](https://github.com/kymatio/kymatio/blob/master/CODE_OF_CONDUCT.md)


### Building the documentation from source
The documentation can also be found in the `doc/` subfolder of the GitHub repository.
To build the documentation locally, please clone this repository and run

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

## About Us

### Core contributors

We are a team of students and researchers at various academic institutions: Berkeley, CNRS, Cornell, ENPC, ENS, Flatiron Institute, NYU, U of Montreal, and WWU; to name a few.

For more information, please see [the full list of contributors](https://github.com/kymatio/kymatio/graphs/contributors).


### Institutional support

We wish to thank the Scientific Computing Core at the Flatiron Institute for the use of their computing resources for testing.

[![Flatiron](doc/source/_static/FL_Full_Logo_Mark_Small.png)](https://www.simonsfoundation.org/flatiron)

We would also like to thank École Normale Supérieure for their support.

[![ENS](https://www.ens.fr/sites/default/files/inline-images/logo.jpg)](https://www.ens.fr/)


### What Kymatio means

Kyma (*κύμα*) means *wave* in Greek. By the same token, Kymatio (*κυμάτιο*) means *wavelet*.

Note that the organization and the library are capitalized (*Kymatio*) whereas the corresponding Python module is written in lowercase (`import kymatio`).

The recommended pronunciation for Kymatio is *kim-ah-tio*. In other words, it rhymes with patio, not with ratio.
