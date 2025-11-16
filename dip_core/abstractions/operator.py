
from abc import ABC, abstractmethod
import torch

class Operator(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward model used inside the loss.
        """
        pass
    