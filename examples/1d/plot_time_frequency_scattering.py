"""
Joint time-frequency scattering of a synthetic signal
=====================================================
The joint time-frequency scattering transform (JTFS) extends the 1D scattering
transform by applying a second, *two-dimensional* wavelet transform jointly
over time and log-frequency. This makes it sensitive to the **direction** of
frequency modulation, which ordinary time scattering discards. In this example
we build a signal containing an ascending chirp followed by a descending chirp
and inspect its JTFS coefficients in both output formats.
"""

###############################################################################
# Import the necessary packages
# -----------------------------
from kymatio.numpy import TimeFrequencyScattering
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
# Generate a signal with two chirps of opposite direction
# --------------------------------------------------------
# JTFS responds differently to upward and downward frequency sweeps, so we
# build a signal whose first half sweeps upward in frequency and whose second
# half sweeps downward.
N = 2 ** 13
t = np.arange(N) / N


def chirp(f0, f1, time):
    """A linear frequency sweep from ``f0`` to ``f1`` (normalized frequency)."""
    instantaneous = f0 * time + 0.5 * (f1 - f0) * time ** 2
    return np.cos(2 * np.pi * instantaneous * N)


x = np.zeros(N, dtype='float32')
half = N // 2
x[:half] = chirp(0.02, 0.2, t[:half])        # ascending sweep
x[half:] = chirp(0.2, 0.02, t[:N - half])    # descending sweep
x *= np.hanning(N).astype('float32')

plt.figure(figsize=(8, 2))
plt.plot(x)
plt.title("Signal: an ascending chirp followed by a descending chirp")

###############################################################################
# Spectrogram
# -----------
# The spectrogram shows the two opposite frequency sweeps.
plt.figure(figsize=(8, 4))
plt.specgram(x, Fs=N)
plt.title("Spectrogram")

###############################################################################
# Time-format JTFS
# ----------------
# With ``format='time'`` the output is a 3D array
# ``(batch, n_coefficients, time)`` in which every joint path is flattened
# onto the coefficient axis. We add a leading batch axis to the signal.
J = 8
Q = 8
J_fr = 3

jtfs_time = TimeFrequencyScattering(J=J, J_fr=J_fr, Q=Q, shape=(N,),
                                    format='time')
Sx_time = jtfs_time(x[None, :])
print("format='time' output shape:", Sx_time.shape)

plt.figure(figsize=(8, 4))
plt.imshow(Sx_time[0], aspect='auto')
plt.title("JTFS coefficients (format='time')")
plt.xlabel("Time")
plt.ylabel("Joint path")

###############################################################################
# Joint-format JTFS
# -----------------
# With ``format='joint'`` the frequency axis is kept explicit, giving a 4D
# array ``(batch, n_jtfs, n_freq, time)``. Each of the ``n_jtfs`` slices is a
# localized time-frequency image; the ``meta()`` dictionary records the
# parameters (including the *spin* ``s = +1``/``-1`` that distinguishes upward
# from downward modulation) of every slice.
jtfs_joint = TimeFrequencyScattering(J=J, J_fr=J_fr, Q=Q, shape=(N,),
                                     format='joint')
Sx_joint = jtfs_joint(x[None, :])
print("format='joint' output shape:", Sx_joint.shape)

###############################################################################
# We display a handful of second-order joint slices as time-frequency images.
n_show = 6
fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 3))
for i, ax in enumerate(axes):
    ax.imshow(Sx_joint[0, i], aspect='auto')
    ax.set_title("path {}".format(i))
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("JTFS joint slices, each (n_freq x time)  -  format='joint'")
plt.tight_layout()
plt.show()
