kymatio: wavelet scattering in PyTorch
======================================

kymatio is a Python module for wavelets and scattering transforms, built on top of PyTorch.

[![PyPI](https://img.shields.io/badge/python-3.6-blue.svg)]()
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/kymatio/kymatio.svg?branch=master)](https://travis-ci.org/kymatio/kymatio)


Use kymatio if you need a library that:
* integrates wavelet scattering in a deep learning architecture,
* supports 1-D, 2-D, and 3-D wavelets, and
* runs seamlessly on CPU and GPU hardware.

Website: [kymatio.github.io](https://kymatio.github.io)


## Installation

### Dependencies

kymatio requires:

* Python (>= 3.6)
* PyTorch (>= 0.4)
* SciPy (>= 0.13)



### Linux

```
conda install pytorch torchvision -c pytorch
pip install -i https://test.pypi.org/simple/ kymatio==0.0.1
```


### Mac OS X

```
conda install pytorch torchvision -c pytorch
pip install -i https://test.pypi.org/simple/ kymatio==0.0.1
```



The software was tested on Linux with anaconda Python 3 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

The software uses PyTorch + NumPy FFT on CPU, and PyTorch + CuPy + CuFFT on GPU.


If using this code for your research please cite our paper:

The scattering authors, [*kymatio: Fast Scattering in 1-D,2-D,3-D*]()

This code unifies multiple previous efforts:
    - pyscatwave/scatwave
    - scatnetlight 
    - others

### Optimized package

After installing the standard package and if you have a CUDA friendly machine you can use:

```
pip install -r requirements_optional_cuda.txt
```

This will allow you to use the optimized backend `skcuda`, as explained in the related documentation.
    

## Usage

Example:

```python
import torch
from kymatio.scattering2d import Scattering2D

scattering = Scattering2D(M=32, N=32, J=2).cuda()
x = torch.randn(1, 3, 32, 32).cuda()

print (scattering(x).size())
```

## Benchmarks
We do some simple timings and comparisons to the previous (multi-core CPU) implementation of scattering (ScatnetLight). We benchmark the software using a 1080 GPU. Below we show input sizes (WxHx3xBatchSize) and speed:

32 × 32 × 3 × 128 (J=2)- 0.03s (speed of 8x vs ScatNetLight)

256 × 256 × 3 × 128 (J=2) - 0.71 s (speed up of 225x vs ScatNetLight)


## Documentation

For building documentation, in the main folder, please do:

```
pip install -r requirements_optional.txt
cd doc; make clean; make html
```

Then, you can read the documentation from `doc/build/html/index.html`.

## Contribution

Contributions are welcome via the standard practices of OSS development: Open a PR to address an open issue, or open an issue to inform us of problems you have experienced or enhancements you would like to propose. Good practices are explained in the scikit-learn documentation.




