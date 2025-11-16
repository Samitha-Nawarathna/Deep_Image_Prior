
from abc import ABC, abstractmethod

class LossFactory(ABC):
    @abstractmethod
    def create_loss(self):
        """
        Return a loss function (callable).
        """
        pass
    