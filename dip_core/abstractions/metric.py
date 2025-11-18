from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract base for a metric function."""
    
    @abstractmethod
    def __call__(self, **kwargs):
        """
        Calculate the metric.
        kwargs can contain 'gt', 'output', 'mask', etc.
        Returns a float or scalar tensor.
        """
        pass
    
    @abstractmethod
    def name(self):
        """Return the name of the metric for logging."""
        pass
