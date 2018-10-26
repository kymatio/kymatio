"""
Matplotlib colormaps in Nilearn
================================
Visualize HCP connectome workbench color maps shipped with Nilearn
which can be used for plotting brain images on surface.
See :ref:`surface-plotting` for surface plotting details.
"""
import numpy as np
import matplotlib.pyplot as plt
from scattering.scattering2d.filters_bank import filters_bank
import scipy.fftpack as fft
import numpy as np


###########################################################################
# Plot color maps
# ----------------

M = 32
J = 3
L = 8

filters_set = filters_bank(M, M, J, L=L)

fig, axs = plt.subplots(J+1, L, sharex=True, sharey=True)


i=0
from colorsys import hls_to_rgb


def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B =  1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

for filter in filters_set['psi']:

    f_r = filter[0][...,0].numpy()
    f_i = filter[0][..., 1].numpy()
    f = f_r + 1j*f_i
    filter_c = fft.fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    phase = np.angle(filter_c)
    amplitude = np.absolute(filter_c)
    #print(amplitude.shape)
    axs[i//L,i%L].imshow(colorize(filter_c))
    axs[i//L,i%L].axis('off')
    #amplitude, cmap=phase, aspect='auto')
    #plt.title('lul', fontsize=10, va='bottom', rotation=90
    i=i+1


f_r = filters_set['phi'][0][...,0].numpy()
f_i = filters_set['phi'][0][..., 1].numpy()
f = f_r + 1j*f_i

filter_c = fft.fft2(f)
filter_c = np.fft.fftshift(filter_c)

axs[J,L//2].imshow(colorize(filter_c))#amplitude, cmap=phase, aspect='auto')

plt.show()

"""
nmaps = len(nilearn_cmaps)
a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
# Initialize the figure
plt.figure(figsize=(10, 4.2))
plt.subplots_adjust(top=0.4, bottom=0.05, left=0.01, right=0.99)
for index, cmap in enumerate(nilearn_cmaps):
    plt.subplot(1, nmaps + 1, index + 1)
    plt.imshow(a, cmap=nilearn_cmaps[cmap])
    plt.axis('off')
    plt.title(cmap, fontsize=10, va='bottom', rotation=90)
###########################################################################
# Plot matplotlib color maps
# --------------------------
plt.figure(figsize=(10, 5))
plt.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)
deprecated_cmaps = ['Vega10', 'Vega20', 'Vega20b', 'Vega20c', 'spectral']
m_cmaps = []
for m in plt.cm.datad:
    if not m.endswith("_r") and m not in deprecated_cmaps:
        m_cmaps.append(m)
m_cmaps.sort()
for index, cmap in enumerate(m_cmaps):
    plt.subplot(1, len(m_cmaps) + 1, index + 1)
    plt.imshow(a, cmap=plt.get_cmap(cmap), aspect='auto')
    plt.axis('off')
    plt.title(cmap, fontsize=10, va='bottom', rotation=90)
show()"""