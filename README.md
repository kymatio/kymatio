kymatio
==========

CuPy/PyTorch implementation of the Scattering transform for 1-D,2-D,3-D.

A scattering network is a Convolutional Network with filters predefined to be wavelets that are not learned and it can be used in vision task such as classification of images, sounds, and molecules. 

Features:
- Integration of 1-D,2-D, and 3-D scattering transforms on gpu
- Easily integrated with deep learning pipelines, especially pytorch. 
- 


The software uses PyTorch + NumPy FFT on CPU, and PyTorch + CuPy + CuFFT on GPU.





If using this code for your research please cite our paper:

M. Andreux, M. Eickenberg, Georgios, E. Oyallon, Joakim,Vincent,  E. Belilovsky, S. Zagoruyko, J. Bruna S. Mallat [*kymatio: Fast Scattering in 1-D,2-D,3-D*]()

This code unifies multiple previous efforts:
    - pyscatwave/scatwave
    - scatnetlight 
    - others 
    

## Installation

The software was tested on Linux with anaconda Python 2.7 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

The first step is to install pytorch following instructions from
<http://pytorch.org>, then you can run `pip`:


```
sudo apt-get install libfftw3-dev
pip install -r requirements.txt
python setup.py install
```

## Usage

Example:

```python
import torch
from scattering.scattering import Scattering

scat = Scattering(M=32, N=32, J=2).cuda()
x = torch.randn(1, 3, 32, 32).cuda()

print scat(x).size()
```

## Benchmarks
We do some simple timings and comparisons to the previous (multi-core CPU) implementation of scattering (ScatnetLight). We benchmark the software using a 1080 GPU. Below we show input sizes (WxHx3xBatchSize) and speed:

32 × 32 × 3 × 128 (J=2)- 0.03s (speed of 8x vs ScatNetLight)

256 × 256 × 3 × 128 (J=2) - 0.71 s (speed up of 225x vs ScatNetLight)




## Contribution

All contributions are welcome.


