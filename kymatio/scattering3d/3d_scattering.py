from frontend import HarmonicScattering3D
import torch
import numpy as np
import matplotlib.pyplot as plt
S = HarmonicScattering3D(3, (64,64,64))
x =  torch.zeros((1, 64,64,64))
x[:, 16:48, 16:48, 16:48] = 1
print(x[0,32, ::4, ::4])
print(x)

Sx = S(x)

print(Sx.size())
print(Sx)
Sx = Sx.detach().cpu().numpy()
norms = np.linalg.norm(Sx.reshape(Sx.shape[:-3] + (-1,)), axis=-1)
Sx = Sx/(1+norms[...,np.newaxis,np.newaxis,np.newaxis])

padded = np.pad(Sx, [(0, 0), (0,0), (0, 0), (1, 1), (1,1)], mode="constant",
        constant_values=np.nan)


panels = padded.reshape(1, 37, 2, 4, 10, 10).transpose(0,1,2,4,3,5).reshape(1,
        37, 2*10, 4*10)
padded_panels = np.pad(panels, [(0, 0), (0, 0), (1, 1), (1,1)]
        , mode="constant", constant_values=np.nan)

panel = padded_panels[0,1:].reshape(6,6,22,42).transpose(0, 2, 1,
        3).reshape(6*22, 6*42)
plt.figure(figsize=(10, 10))
plt.imshow(panel, cmap='gray')
plt.savefig("panel.png")
