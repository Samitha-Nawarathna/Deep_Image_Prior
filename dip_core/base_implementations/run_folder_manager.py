import os
from datetime import datetime

class RunFolderManager:
    def __init__(self, exp_type, log_dir):
        
        self.base = f"{log_dir}/logs/{exp_type}/"

    def create_run_folder(self, mode="experiment"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if mode == "experiment":
            path = f"{self.base}/runs/{timestamp}"
        else:
            path = f"{self.base}/debug/dbg_{timestamp}"

        os.makedirs(path, exist_ok=True)
        return path

    def save_config(self, config, run_dir):
        with open(os.path.join(run_dir, "config.yml"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")