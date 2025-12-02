import yaml
import copy

class ConfigManager:
    def __init__(self, template_path):
        with open(template_path, "r") as f:
            self.template = yaml.safe_load(f)

    def validate(self, overrides):
        for key in overrides:
            if key not in self.template:
                raise ValueError(f"Invalid config parameter: {key}")

    def merge(self, overrides):
        self.validate(overrides)
        cfg = copy.deepcopy(self.template)
        cfg.update(overrides)
        return cfg

    def load(self, concrete_config_path):
        with open(concrete_config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config

    def save(self, cfg, path):
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
