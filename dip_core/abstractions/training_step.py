
from abc import ABC, abstractmethod

class TrainingStep(ABC):
    @abstractmethod
    def __call__(self, model, input_tensor, target_tensor,
                 operator, loss_fn, optimizer, logger, iteration):
        """
        Perform a single optimization step.
        """
        pass
    