# This collection of tests checks the correspondence between numpy and jax
# for the functionality relevant to kymatio. In particular the fft routines
# have different backends, which leads to relatively poor numerical accuracy
# (rel err around 1e-4). We bound this here and will thus be notified if
# a change to the library makes it worse.

import pytest
import numpy as np
import scipy.fftpack

import jax.numpy as jnp
from jax import random, device_put, config

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
    real_filter = np.random.rand(1024)
    complex_input = np.random.rand(4, 1, 1024) + 1j * np.random.rand(4, 1, 1024)

    numpy_result = complex_input * real_filter

    real_filter = device_put(jnp.asarray(real_filter))
    complex_input = device_put(jnp.asarray(complex_input))

    jax_result = complex_input * real_filter
    assert np.allclose(jax_result, numpy_result)

def test_real_ifft():
    x = np.random.rand(4, 1, 1024)

    numpy_x_ifft = np.real(scipy.fftpack.ifft(x))

    jax_x_ifft = jnp.real(jnp.fft.ifft(device_put(jnp.asarray(x))))
    assert np.allclose(jax_x_ifft, numpy_x_ifft, atol=1e-6, rtol=1e-7)

def test_jnp_real_ifft():
    x = device_put(random.normal(key, (4, 1, 1024)))

    jax_x_ifft = jnp.real(jnp.fft.ifft(device_put(x)))

    numpy_x_ifft = np.real(scipy.fftpack.ifft(np.asarray(x)))
    assert jnp.allclose(numpy_x_ifft, jax_x_ifft, atol=1e-6, rtol=1e-7)

def test_ifft():
    x = np.random.rand(4, 1, 1024)

    numpy_x_ifft = scipy.fftpack.ifft(x)

    jax_x_ifft = jnp.fft.ifft(device_put(jnp.asarray(x)))
    assert np.allclose(jax_x_ifft, numpy_x_ifft, atol=1e-6, rtol=1e-7)

def test_jnp_ifft():
    x = device_put(random.normal(key, (4, 1, 1024)))

    jax_x_ifft = jnp.fft.ifft(device_put(x))

    numpy_x_ifft = scipy.fftpack.ifft(np.asarray(x))
    assert jnp.allclose(numpy_x_ifft, jax_x_ifft, atol=1e-6, rtol=1e-7)

def test_fft():
    real_filter = np.random.rand(1024)
    complex_input = np.random.rand(4, 1, 1024) + 1j * np.random.rand(4, 1, 1024)

    numpy_result = np.fft.fft(complex_input * real_filter)

    real_filter = device_put(jnp.asarray(real_filter))
    complex_input = device_put(jnp.asarray(complex_input))

    jax_result = jnp.fft.fft(complex_input * real_filter)
    assert np.allclose(jax_result, numpy_result, atol=1e-4, rtol=1e-4)

def test_jnp_fft():
    real_filter = device_put(random.normal(key, (1024,)))
    complex_input = device_put(random.normal(key, (4, 1, 1024)) + 1j * random.normal(key, (4, 1, 1024)))

    jax_result = jnp.fft.fft(complex_input * real_filter)

    real_filter = np.asarray(real_filter)
    complex_input = np.asarray(complex_input)

    numpy_result = np.fft.fft(complex_input * real_filter)

    assert jnp.allclose(numpy_result, jax_result, atol=1e-3, rtol=1e-4)

def test_rfft():
    x = np.random.rand(4, 1, 1024)
    numpy_result = np.fft.fft(x)

    x = device_put(jnp.asarray(x))
    jax_result = jnp.fft.fft(x)

    assert np.allclose(jax_result, numpy_result, atol=1e-4, rtol=1e-4)

def test_jnp_rfft():
    x = device_put(random.normal(key, (4, 1, 1024)))
    jax_result = jnp.fft.fft(x)

    x = np.asarray(x)
    numpy_result = np.fft.fft(x)

    assert jnp.allclose(numpy_result, jax_result, atol=1e-4, rtol=1e-4)


def test_absolute():
    complex_input = np.random.rand(4, 1, 1024) + 1j * np.random.rand(4, 1, 1024)
    numpy_result = np.absolute(complex_input)

    complex_input = device_put(jnp.asarray(complex_input))
    jax_result = jnp.absolute(complex_input)
    assert np.allclose(jax_result, numpy_result)

def test_jnp_absolute():
    complex_input = device_put(random.normal(key, (4, 1, 1024)) + 1j * random.normal(key, (4, 1, 1024)))
    jax_result = jnp.absolute(complex_input)

    complex_input = np.asarray(complex_input)
    numpy_result = np.absolute(complex_input)
    assert jnp.allclose(numpy_result, jax_result)

def test_real():
    complex_input = np.random.rand(4, 1, 1024) + 1j * np.random.rand(4, 1, 1024)
    numpy_result = jnp.real(complex_input)

    complex_input = device_put(jnp.asarray(complex_input))
    jax_result = jnp.real(complex_input)
    assert np.allclose(jax_result, numpy_result)

def test_jnp_real():
    complex_input = device_put(random.normal(key, (4, 1, 1024)) + 1j * random.normal(key, (4, 1, 1024)))
    jax_result = jnp.real(complex_input)

    complex_input = np.asarray(complex_input)
    numpy_result = np.real(complex_input)
    assert jnp.allclose(numpy_result, jax_result)

def test_subsample_fourier():
    x = np.random.rand(4, 1, 1024)
    x_jax = device_put(jnp.asarray(x))

    for subsampling_factor in [1, 2, 4]:
        numpy_result = subsample_fourier(x, subsampling_factor)
        jax_result = subsample_fourier(x_jax, subsampling_factor)
        assert np.allclose(jax_result, numpy_result)
        assert jnp.allclose(numpy_result, jax_result)


def test_pad_unpad():       
    x = np.random.rand(4, 1, 512)
    numpy_result = np_pad(x, 256, 256)

    x = device_put(jnp.asarray(x))
    jax_result = jnp_pad(x, 256, 256)

    assert np.allclose(jax_result, numpy_result)

    numpy_result = numpy_result[..., 4: 12]
    jax_result = jax_result[..., 4: 12]
    assert np.allclose(jax_result, numpy_result)

def test_jnp_pad_unpad():       
    x = device_put(random.normal(key, (4, 1, 512)))
    jax_result = jnp_pad(x, 256, 256)

    x = np.asarray(x)
    numpy_result = np_pad(x, 256, 256)
    assert jnp.allclose(numpy_result, jax_result)

    jax_result = jax_result[..., 4 : 12]
    numpy_result = numpy_result[..., 4 : 12]
    assert jnp.allclose(numpy_result, jax_result)

def test_stack():       
    array_1 = np.random.rand(4, 1, 512)
    array_2 = np.random.rand(4, 1, 512)
    numpy_result = np.stack([array_1, array_2], axis=-2)

    array_1 = device_put(jnp.asarray(array_1))
    array_2 = device_put(jnp.asarray(array_2))
    jax_result = jnp.stack([array_1, array_2], axis=-2)

    assert np.allclose(jax_result, numpy_result)

def test_jnp_stack():       
    array_1 = device_put(random.normal(key, (4, 1, 512)))
    array_2 = device_put(random.normal(key, (4, 1, 512)))
    jax_result = jnp.stack([array_1, array_2], axis=-2)

    array_1 = np.asarray(array_1)
    array_2 = np.asarray(array_2)
    numpy_result = np.stack([array_1, array_2], axis=-2)

    assert jnp.allclose(numpy_result, jax_result)

