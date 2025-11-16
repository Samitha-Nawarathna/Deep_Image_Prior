
from abc import ABC, abstractmethod

class Stopper(ABC):
    @abstractmethod
    def should_stop(self, iteration, metrics) -> bool:
        """
        Decide when to stop training.
        """
        pass
    