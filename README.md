kymatio: wavelet scattering in PyTorch
======================================

kymatio is a Python module for wavelet and scattering transforms, built on top of PyTorch.

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/kymatio/kymatio.svg?branch=master)](https://travis-ci.org/kymatio/kymatio)


Use kymatio if you need a library that:
* integrates wavelet scattering in a deep learning architecture,
* supports 1-D, 2-D, and 3-D wavelets, and
* runs seamlessly on CPU and GPU hardware.

Website: [https://kymatio.github.io](https://kymatio.github.io)


## Installation

### Dependencies

kymatio requires:

* Python (>= 3.6)
* Anaconda
* PyTorch (>= 0.4)
* SciPy (>= 0.13)



### Linux

```
conda install pytorch torchvision -c pytorch
pip install -i https://test.pypi.org/simple/ kymatio==0.0.1
```


### macOS

```
conda install pytorch torchvision -c pytorch
pip install -i https://test.pypi.org/simple/ kymatio==0.0.1
```



The software was tested on Linux with Anaconda Python 3 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

The software uses PyTorch + NumPy FFT on CPU, and PyTorch + CuPy + CuFFT on GPU.


If you use this code in your work please cite our paper:

The scattering authors, [*kymatio: Fast Scattering in 1-D,2-D,3-D*]()

This code unifies multiple previous efforts:
    - PyScatWave/ScatWave,
    - ScatNetLight,
    - ScatNet, and others

### Optimized package

If you have a CUDA-enabled GPU, you may run

```
pip install -r requirements_optional_cuda.txt
```

after installation to install the optimized `skcuda` backend. To enable it, set
the `KYMATIO_BACKEND` environment variable to `skcuda`. For more information,
see the documentation.


## Documentation

To build the documentation, please run

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

You may then read the documentation in `doc/build/html/index.html`.
