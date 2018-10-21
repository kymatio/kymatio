Quick Start
===========

Do `pip install scattering`

Then

```python
from scattering import Scattering1D

```

Or
```python
import torch
from scattering import Scattering2D
scattering = Scattering2D()
signal = torch.randn(1, 32, 32)
output = scattering(signal)
```


