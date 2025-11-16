import os

# -----------------------------------------------------------
# Helper function to write files safely
# -----------------------------------------------------------
def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -----------------------------------------------------------
# Folder and file definitions
# -----------------------------------------------------------

files = {

    # =======================
    # ABSTRACT CLASSES
    # =======================
    "dip_core/abstractions/trainer.py":
    """
from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def train(self):
        \"""
        Run the full training loop.
        \"""
        pass
    """,

    "dip_core/abstractions/training_step.py":
    """
from abc import ABC, abstractmethod

class TrainingStep(ABC):
    @abstractmethod
    def __call__(self, model, input_tensor, target_tensor,
                 operator, loss_fn, optimizer, logger, iteration):
        \"""
        Perform a single optimization step.
        \"""
        pass
    """,

    "dip_core/abstractions/network_factory.py":
    """
from abc import ABC, abstractmethod

class NetworkFactory(ABC):
    @abstractmethod
    def create_network(self):
        \"""
        Return a torch.nn.Module instance for DIP.
        \"""
        pass
    """,

    "dip_core/abstractions/loss_factory.py":
    """
from abc import ABC, abstractmethod

class LossFactory(ABC):
    @abstractmethod
    def create_loss(self):
        \"""
        Return a loss function (callable).
        \"""
        pass
    """,

    "dip_core/abstractions/optimizer_factory.py":
    """
from abc import ABC, abstractmethod

class OptimizerFactory(ABC):
    @abstractmethod
    def create_optimizer(self, model_parameters):
        \"""
        Return an optimizer instance.
        \"""
        pass
    """,

    "dip_core/abstractions/logger.py":
    """
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log_scalar(self, name, value, step):
        pass

    @abstractmethod
    def log_image(self, name, image_tensor, step):
        pass

    @abstractmethod
    def save_config(self, config: dict):
        pass
    """,

    "dip_core/abstractions/operator.py":
    """
from abc import ABC, abstractmethod
import torch

class Operator(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"""
        Forward model used inside the loss.
        \"""
        pass
    """,

    "dip_core/abstractions/stopper.py":
    """
from abc import ABC, abstractmethod

class Stopper(ABC):
    @abstractmethod
    def should_stop(self, iteration, metrics) -> bool:
        \"""
        Decide when to stop training.
        \"""
        pass
    """,

    # =======================
    # BASE IMPLEMENTATIONS
    # =======================

    "dip_core/base_implementations/simple_logger.py":
    """
import os
import json
import numpy as np
from PIL import Image
from dip_core.abstractions.logger import Logger

class SimpleLogger(Logger):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def save_config(self, config):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def log_scalar(self, name, value, step):
        with open(os.path.join(self.log_dir, f"{name}.txt"), "a") as f:
            f.write(f"{step}: {value}\\n")

    def log_image(self, name, image_tensor, step):
        arr = (
            image_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255
        ).astype(np.uint8)
        Image.fromarray(arr).save(
            os.path.join(self.log_dir, f"{name}_{step}.png")
        )
    """,

    "dip_core/base_implementations/default_optimizer.py":
    """
import torch
from dip_core.abstractions.optimizer_factory import OptimizerFactory

class DefaultOptimizerFactory(OptimizerFactory):
    def __init__(self, lr):
        self.lr = lr

    def create_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.lr)
    """,

    "dip_core/base_implementations/default_operator.py":
    """
from dip_core.abstractions.operator import Operator

class IdentityOperator(Operator):
    def forward(self, x):
        return x
    """,

    "dip_core/base_implementations/default_stopper.py":
    """
from dip_core.abstractions.stopper import Stopper

class MaxIterationStopper(Stopper):
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def should_stop(self, iteration, metrics):
        return iteration >= self.max_iter
    """,

    # =======================
    # UTILITIES
    # =======================

    "dip_core/utils/seeds.py":
    """
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    """,

    "dip_core/utils/image_io.py":
    """
from PIL import Image
import numpy as np
import torch

def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0)

def save_image(tensor, path):
    arr = (
        tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    Image.fromarray(arr).save(path)
    """,

    "dip_core/utils/metrics.py":
    """
import torch
import torch.nn.functional as F

def mse(x, y):
    return F.mse_loss(x, y)

def psnr(x, y, max_val=1.0):
    mse_val = mse(x, y).item()
    return 20 * torch.log10(max_val / (mse_val ** 0.5 + 1e-8))
    """,

    "dip_core/utils/plotting.py":
    """
import matplotlib.pyplot as plt

def plot_curve(values, save_path):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(save_path)
    plt.close()
    """,
}

# -----------------------------------------------------------
# Write all files
# -----------------------------------------------------------
for path, content in files.items():
    write(path, content)

print("✅ DIP core structure generated successfully!")
