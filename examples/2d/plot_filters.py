"""
Matplotlib colormaps in Nilearn
================================
Visualize HCP connectome workbench color maps shipped with Nilearn
which can be used for plotting brain images on surface.
See :ref:`surface-plotting` for surface plotting details.
"""
import numpy as np
import matplotlib.pyplot as plt
from scattering.scattering2d.filter_bank import scattering_filter_factory_real


###########################################################################
# Plot color maps
# ----------------

M = 128
J = 3

filters_set = scattering_filter_factory_real(M, M, J)

for filter in filters_set['psi']:
    filter_c = fft.fft2(filter[0])
    phase = np.angle(filter_c)
    amplitude = np.absolute(filter_c)

    plt.imshow(amplitude, cmap=plt.get_cmap(cmap), aspect='auto')
    plt.title('lul', fontsize=10, va='bottom', rotation=90)

show()

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