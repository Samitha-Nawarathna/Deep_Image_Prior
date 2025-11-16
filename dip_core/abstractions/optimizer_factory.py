
from abc import ABC, abstractmethod

class OptimizerFactory(ABC):
    @abstractmethod
    def create_optimizer(self, model_parameters):
        """
        Return an optimizer instance.
        """
        pass
    