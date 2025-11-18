import torch
from dip_core.abstractions.trainer import Trainer
from tqdm import trange

class DefaultTrainer(Trainer):
    def __init__(self, model, training_step, operator, stopper, optimizer, loss_fn, noise, target, logger, config, image_per = 1):
        self.model = model
        self.step_fn = training_step
        self.operator = operator
        self.stopper = stopper
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.noise = noise
        self.target = target
        self.logger = logger
        self.config = config
        self.image_per = image_per

    def train(self):
        for i in trange(1, self.config["iterations"] + 1):
            metrics = self.step_fn(
                self.model, self.noise, self.target,
                self.operator, self.loss_fn, self.optimizer,
                self.logger, iteration=i
            )

            self.logger.log_results(metrics, i)

            if self.stopper.should_stop(i, metrics):
                break
