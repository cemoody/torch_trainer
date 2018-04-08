Very simple pytorch training utility.

- Allows both `forward` and `loss` functions to access model parameters
- Can register callback functions

```
from trainer import Trainer
from callbacks import auc_callback

class Model(nn.Module):
    def __init__(self, lam=0.1):
        self.lin = nn.Linear(1, 1)
        self.lam = lam

    def forward(self, x):
        return self.lin(x)

    def loss(self, prediction, target):
        llh = ((prediction - target)**2.0).sum()
        reg = self.lin.weight.sum()
        return llh + reg * self.lam


model = Model()


callbacks = {'auc': auc_callback}
t = Trainer(model, optimizer, batchsize=128,
            callbacks=callbacks, seed=42)
train = np.concatenate((X, y))
t.fit(train)
```
