import numpy as np
import matplotlib.pyplot as plt
from scattering.scattering2d.filters_bank import filters_bank
import scipy.fftpack as fft
import numpy as np
from matplotlib import rc


# This function displays filter bank with those parameters:
M = 32
J = 3
L = 8

filters_set = filters_bank(M, M, J, L=L)

fig, axs = plt.subplots(J+1, L, sharex=True, sharey=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
i=0

# Thanks to https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
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

# Display first the psi
for filter in filters_set['psi']:
    f_r = filter[0][...,0].numpy()
    f_i = filter[0][..., 1].numpy()
    f = f_r + 1j*f_i
    filter_c = fft.fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    axs[i//L,i%L].imshow(colorize(filter_c))
    axs[i//L,i%L].axis('off')
    axs[i//L,i%L].set_title("$j = "+str(i//L)+"$ \n $\\theta="+str(i%L)+"$")
    i = i+1


# Add blanks for pretty display
for z in range(L):
    axs[i // L, i % L].axis('off')
    i = i+1


# Display the phi
f_r = filters_set['phi'][0][...,0].numpy()
f_i = filters_set['phi'][0][..., 1].numpy()
f = f_r + 1j*f_i
filter_c = fft.fft2(f)
filter_c = np.fft.fftshift(filter_c)
axs[J,L//2].imshow(colorize(filter_c))#amplitude, cmap=phase, aspect='auto')
fig.suptitle("Wavelets for each scales $j$ and angles $\\theta$ used, with the corresponding low-pass filter.\n The contrast corresponds to the amplitude and the color to the phase.", fontsize=13)


plt.show()