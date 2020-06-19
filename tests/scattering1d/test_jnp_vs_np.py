import pytest
import numpy as np
import scipy.fftpack

import jax.numpy as jnp
from jax import random, device_put
import jax.config as config 

key = random.PRNGKey(0)


def subsample_fourier(x, k):
  N = x.shape[-1]
  res = x.reshape(x.shape[:-1] + (k, N // k)).mean(axis=(-2,))
  return res

def np_pad(x, pad_left, pad_right):
  res = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right),), mode='reflect')
  return res

def jnp_pad(x, pad_left, pad_right):
  res = jnp.pad(x, ((0, 0), (0, 0), (pad_left, pad_right),), mode='reflect')
  return res
  
def test_multiplication():
    # multiplication
    a = np.random.rand(4, 1, 1024)
    b = np.random.rand(4, 1, 1024)
    d = np.random.rand(1024)

    c = a + 1j*b
    e = c*d
    l = e

    a = device_put(jnp.asarray(a))
    b = device_put(jnp.asarray(b))
    d = device_put(jnp.asarray(d))

    c = a + 1j*b
    e = c*d
    assert jnp.allclose(e, l)


def test_jnp_multiplication():
    # multiplication jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    b = device_put(random.normal(key, (4, 1, 1024)))
    d = device_put(random.normal(key, (1024,)))
    c = a + 1j*b
    e = c*d
    l = e
    a = np.asarray(a)
    b = np.asarray(b)
    d = np.asarray(d)
    c = a + 1j*b
    e = c*d
    assert jnp.allclose(e, l)

def test_real_ifft():
    #real ifft
    a = np.random.rand(4, 1, 1024)
    f = np.real(scipy.fftpack.ifft(a))
    l = f
    f = jnp.real(jnp.fft.ifft(device_put(jnp.asarray(a))))
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

def test_jnp_real_ifft():
    #real ifft jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    f = jnp.real(jnp.fft.ifft(device_put(a)))
    l = f
    f = np.real(scipy.fftpack.ifft(np.asarray(a)))
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)
        
def test_ifft():
    #ifft
    a = np.random.rand(4, 1, 1024)
    f = scipy.fftpack.ifft(a)
    l = f
    f = jnp.fft.ifft(device_put(jnp.asarray(a)))
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

def test_jnp_ifft():
    #ifft jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    f = jnp.fft.ifft(device_put(a))
    l = f
    f = scipy.fftpack.ifft(np.asarray(a))
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

def test_fft():
    #fft
    a = np.random.rand(4, 1, 1024)
    b = np.random.rand(4, 1, 1024)
    d = np.random.rand(1024)
    c = a + 1j*b
    e = d*c
    f = np.fft.fft(e)
    l = f
    
    a = device_put(jnp.asarray(a))
    b = device_put(jnp.asarray(b))
    c = a + 1j*b
    d = device_put(jnp.asarray(d))
    e = d*c
    f = jnp.fft.fft(e)
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

def test_jnp_fft():
    #fft jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    b = device_put(random.normal(key, (4, 1, 1024)))
    c = a + 1j*b
    l = jnp.fft.fft(c)
    a = np.asarray(a)
    b = np.asarray(b)
    c = a + 1j*b
    f = np.fft.fft(c)
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

def test_rfft():
    #rfft
    a = np.random.rand(4, 1, 1024)
    l = np.fft.fft(a)
    
    a = device_put(jnp.asarray(a))
    f = jnp.fft.fft(a)
    
    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)
   
def test_jnp_rfft():
    #rfft jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    f = jnp.fft.fft(a)
    
    a = np.asarray(a)
    l = np.fft.fft(a)

    assert jnp.allclose(f, l, atol=1e-6, rtol=1e-7)

    
def test_absolute():
    #absolute 
    a = np.random.rand(4, 1, 1024)
    b = np.random.rand(4, 1, 1024)
    c = a + 1j*b
    l = np.absolute(c)
    a = device_put(jnp.asarray(a))
    b = device_put(jnp.asarray(b))
    c = a + 1j*b
    f = jnp.absolute(c)
    assert jnp.allclose(f, l)
    
def test_jnp_absolute():
    #absolute jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    b = device_put(random.normal(key, (4, 1, 1024)))
    c = a + 1j*b
    l = jnp.absolute(c)
    a = np.asarray(a)
    b = np.asarray(b)
    c = a + 1j*b
    f = np.absolute(c)
    assert jnp.allclose(f, l)

def test_real():
    #real
    a = np.random.rand(4, 1, 1024)
    b = np.random.rand(4, 1, 1024)
    c = a + 1j*b
    l = jnp.real(c)
    a = device_put(jnp.asarray(a))
    b = device_put(jnp.asarray(b))
    c = a + 1j*b
    f = jnp.real(c)
    assert jnp.allclose(f, l)

def test_jnp_real():
    #real jnp
    a = device_put(random.normal(key, (4, 1, 1024)))
    b = device_put(random.normal(key, (4, 1, 1024)))
    c = a + 1j*b
    l = jnp.real(c)
    a = np.asarray(a)
    b = np.asarray(b)
    c = a + 1j*b
    f = np.real(c)
    assert jnp.allclose(f, l)
    
def test_subsample_fourier():
    #subsample fourier
    a = np.random.rand(4, 1, 1024)
    b = device_put(jnp.asarray(a))
    for i in [1, 2, 4]:
        c = subsample_fourier(a, i)
        d = subsample_fourier(a, i)
        assert jnp.allclose(c, d)

def test_jnp_subsample_fourier():       
    #subsample fourier jnp
    a = np.random.rand(4, 1, 1024)
    b = device_put(jnp.asarray(a))
    for i in [1, 2, 4]:
        c = subsample_fourier(a, i)
        d = subsample_fourier(a, i)
        assert jnp.allclose(c, d)

def test_pad_unpad():       
    #pad
    a = np.random.rand(4, 1, 512)
    b = np_pad(a, 256, 256)
    c = device_put(jnp.asarray(a))
    d = jnp_pad(c, 256, 256)
    assert jnp.allclose(b, d)
    b = b[..., 4: 12]
    d = d[..., 4: 12]
    assert jnp.allclose(b, d)

def test_jnp_pad_unpad():       
    #pad jnp
    a = device_put(random.normal(key, (4, 1, 512)))
    b = jnp_pad(a, 256, 256)
    c = np.asarray(a)
    d = np_pad(c, 256, 256)
    assert jnp.allclose(b, d)
    b = b[..., 4: 12]
    d = d[..., 4: 12]
    assert jnp.allclose(b, d)
    
def test_stack():       
    #stack
    a = np.random.rand(4, 1, 512)
    b = np.random.rand(4, 1, 512)
    c = np.stack([a, b], axis=-2)
    d = device_put(jnp.asarray(a))
    e = device_put(jnp.asarray(b))
    f = jnp.stack([d, e], axis=-2)
    assert jnp.allclose(c, f)
    
def test_jnp_stack():       
    #stack jnp 
    a = device_put(random.normal(key, (4, 1, 512)))
    b = device_put(random.normal(key, (4, 1, 512)))
    c = jnp.stack([a, b], axis=-2)
    a = np.asarray(a)
    b = np.asarray(b)
    f = np.stack([a, b], axis=-2)
    assert jnp.allclose(c, f)
