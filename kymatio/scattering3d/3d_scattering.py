from frontend import HarmonicScattering3D
import torch
import numpy as np
import matplotlib.pyplot as plt

orientations = np.array([(1,0,0), (0, 1,0), (0, 0, 1), (1,1,0), (0, 1,1), (1, 0, 1)])
S = HarmonicScattering3D(3, (64,64,64), orientations=orientations)
x =  torch.zeros((1, 64,64,64))
x[:, 16:48, 16:48, 16:48] = 1
print(x[0,32, ::4, ::4])
print(x)

Sx = S(x)

print(Sx.size())
#print(Sx)
Sx = Sx.detach().cpu().numpy()
norms = np.linalg.norm(Sx.reshape(Sx.shape[:-3] + (-1,)), axis=-1, ord=np.inf)
Sx = Sx/(1e-19+norms[...,np.newaxis,np.newaxis,np.newaxis])

padded = np.pad(Sx, [(0, 0), (0,0), (0, 0), (1, 1), (1,1)], mode="constant",
        constant_values=np.nan)
J= 3
L = 6
order0_size = 1
order1_size = L * J
order2_size = L ** 2 * J * (J - 1) // 2
size = order0_size+order1_size +order2_size

panels = padded.reshape(1, size, 2, 4, 10, 10).transpose(0,1,2,4,3,5).reshape(1,
        size, 2*10, 4*10)
padded_panels = np.pad(panels, [(0, 0), (0, 0), (1, 1), (1,1)]
        , mode="constant", constant_values=np.nan)

panel = padded_panels[0,1:121].reshape(10,12,22,42).transpose(0, 2, 1,
        3).reshape(10*22, 12*42)
plt.figure(figsize=(20, 20), dpi=600)
plt.imshow(panel, cmap='gray')
plt.savefig("panel.png")



