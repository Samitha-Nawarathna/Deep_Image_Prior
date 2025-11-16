
import torch
from dip_core.abstractions.optimizer_factory import OptimizerFactory

class DefaultOptimizerFactory(OptimizerFactory):
    def __init__(self, lr):
        self.lr = lr

    def create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.lr)
    