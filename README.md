Kymatio: wavelet scattering in PyTorch
======================================

Kymatio is an implementation of the wavelet scattering transform in the Python programming language, suitable for large-scale numerical experiments in signal processing and machine learning.

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/kymatio/kymatio.svg?branch=master)](https://travis-ci.org/kymatio/kymatio)


Use Kymatio if you need a library that:
* supports 1-D, 2-D, and 3-D wavelets,
* integrates wavelet scattering in a deep learning architecture, and
* runs seamlessly on CPU and GPU hardware.


## What makes Kymatio different?

Kymatio stands out with respect to these other packages thanks to three assets: flexibility, portability, and scalability.

### Flexibility

The Kymatio organization associates the developers of several pre-existing different packages for wavelet scattering, including `ScatNet`, `scattering.m`, `PyScatWave`, `WaveletScattering.jl`, and `PyScatHarm`.

The resort to PyTorch tensors as inputs to Kymatio allows the programmer to backpropagate the gradient of wavelet scattering coefficients, thus integrating them within an end-to-end trainable pipeline, such as a deep neural network.

### Portability

Each of these algorithms is written in a high-level imperative paradigm, making it portable to any Python library for array operations as long as it enables complex-valued linear algebra and a fast Fourier transform (FFT).

As of the first stable version, NumPy and CuPy are the two available backends, for CPU and GPU hardware respectively.


### Scalability

Kymatio integrates the construction of wavelet filter banks in 1D, 2D, and 3D, as well as memory-efficient algorithms for extracting wavelet scattering coefficients, under a common application programming interface.

Running Kymatio on a graphics processing unit (GPU) rather than a multi-core conventional computer processing unit (CPU) allows to speed up the scattering transform.
As of the alpha release, the speedup with respect to CPU-based MATLAB code is of the order of 10 in 1D and of the order of 100 in 2D.

We refer to our [official benchmarks](https://www.kymat.io/userguide.html#benchmark-with-previous-versions) for further details.


## Installation


### Dependencies

Kymatio requires:

* Python (>= 3.6)
* PyTorch (>= 0.4)
* SciPy (>= 0.13)


### Standard installation (on CPU hardware)
We strongly recommend running Kymatio in a Conda environment, because this simplifies the installation of PyTorch.
One the aformentioned dependencies are installed, you may install the latest version of Kymatio by using the package manager `pip`, which will automatically download Kymatio from the Python Package Index (PyPI):

```
pip install kymatio
```

Linux and macOS are the two operating systems that are officially supported by Kymatio.


### GPU acceleration


To run Kymatio on a graphics processing unit (GPU), you should install the CUDA library and install the scikit-cuda dependency by running the following pip command:

```
pip install scikit-cuda cupy
```

Then, set the `KYMATIO_BACKEND` to `skcuda`:

```
os.environ["KYMATIO_BACKEND"] = "skcuda"
```


#### Available backends: PyTorch and scikit-cuda

Kymatio is designed to interoperate on a variety of backends for array operations.
The user may control the choice of backend at runtime by setting the environment variable `KYMATIO_BACKEND`, or by editing the Kymatio configuration file (`~/.config/kymatio/kymatio.cfg` on Linux).

At the time of alpha release, the two available backends are PyTorch (`torch`) and scikit-cuda (`skcuda`) for 1D scattering and 2D scattering, and PyTorch only for 3D scattering.

PyTorch is the default backend in 1D, 2D, and 3D scattering. Yet, for applications of the 2D scattering transform to large images (e.g. ImageNet, of size 224x224), we recommend the scikit-cuda backend, which is substantially faster than PyTorch.


## Documentation

The documentation of Kymatio is officially hosted on the [kymat.io](https://www.kymat.io/) website.


### Online resources

* [GitHub repository](https://github.com/kymatio/kymatio)
* [GitHub issue tracker](https://github.com/kymatio/kymatio/issues)
* [BSD-3-Clause license](https://github.com/kymatio/kymatio/blob/master/LICENSE.md)
* [List of authors](https://github.com/kymatio/kymatio/blob/master/AUTHORS.md)
* [Code of conduct](https://github.com/kymatio/kymatio/blob/master/CODE_OF_CONDUCT.md)


### Building the documentation from source.
The documentation can also be found in the `doc/` subfolder of the GitHub repository.
To build the documentation locally, please clone this repository and run

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

## Why the name, Kymatio?

Kyma (*κύμα*) means *wave* in Greek. By the same token, Kymatio (*κυμάτιο*) means *wavelet*.

Note that the organization and the library are capitalized (*Kymatio*) whereas the corresponding Python module is written in lowercase (`import kymatio`).

The recommended pronunciation for Kymatio is *kim-ah-tio*. In other words, it rhymes with patio, not with ratio.
