
import os
import json
import numpy as np
from PIL import Image
from dip_core.abstractions.logger import Logger

class DefaultLogger(Logger):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def save_config(self, config):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def log_scalar(self, name, value, step):
        with open(os.path.join(self.log_dir, f"{name}.txt"), "a") as f:
            f.write(f"{step}: {value}\n")

    def log_image(self, name, image_tensor, step):
        arr = (
            image_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255
        ).astype(np.uint8)
        Image.fromarray(arr).save(
            os.path.join(self.log_dir, f"{name}_{step}.png")
        )

    def log_str(self, string, name="gt_results"):
        with open(os.path.join(self.log_dir, f"{name}.txt"), "w") as f:
            json.dump(string, f, indent=4)

    def save_checkpoint(self, model, name="model.pt"):
        path = os.path.join(self.ckpt_folder, name)
        model.save(path)
    