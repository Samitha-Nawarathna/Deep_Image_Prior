
from abc import ABC, abstractmethod

class NetworkFactory(ABC):
    @abstractmethod
    def create_network(self):
        """
        Return a torch.nn.Module instance for DIP.
        """
        pass
    