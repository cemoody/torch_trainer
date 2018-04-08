Very simple pytorch training utility.

- Allows both `forward` and `loss` functions to access model parameters
- Can register callback functions

```python
from torch_trainer.trainer import Trainer
from torch_trainer.callbacks import rms_callback
from torch import nn
from torch.optim import Adam

import numpy as np

class Model(nn.Module):
    def __init__(self, lam=0.1):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.lam = lam

    def forward(self, x1, x2, x3, x4, y):
        return self.lin(x1.unsqueeze(1))

    def loss(self, prediction, x1, x2, x3, x4, y):
        llh = ((prediction - y) **2.0).sum()
        reg = self.lin.weight.sum()
        return llh + reg * self.lam


model = Model()
optim = Adam(model.parameters())

callbacks = {'rms': rms_callback}
t = Trainer(model, optim, batchsize=128,
            callbacks=callbacks, seed=42)
X1, X2, X3, X4 = np.arange(100).reshape((4, 25)).astype('float32')
y = np.arange(25).astype('float32')
t.fit(X1, X2, X3, X4, y)
```
