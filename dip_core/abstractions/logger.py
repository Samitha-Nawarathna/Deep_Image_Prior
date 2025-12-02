
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log_scalar(self, name, value, step):
        pass

    @abstractmethod
    def log_image(self, name, image_tensor, step):
        pass

    @abstractmethod
    def log_str(self, string):
        pass

    @abstractmethod
    def save_config(self, config: dict):
        pass
    