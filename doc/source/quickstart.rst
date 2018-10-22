Quick Start
===========



Do `pip install scattering`

Then

```python
from scattering import Scattering1D

```

Or for 2D signals:

```python
import torch
from scattering import Scattering2D
scattering = Scattering2D(32, 32, 2)
signal = torch.randn(1, 1, 32, 32)
output = scattering(signal)
```

