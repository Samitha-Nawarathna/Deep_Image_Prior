
from dip_core.abstractions.stopper import Stopper

class MaxIterationStopper(Stopper):
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def should_stop(self, iteration, metrics):
        return iteration >= self.max_iter
    