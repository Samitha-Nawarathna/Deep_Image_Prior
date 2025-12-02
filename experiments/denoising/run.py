import os
import yaml
import torch

from dip_core.utils.seeds import set_seed
from dip_core.utils.image_io import load_image
from dip_core.utils.losses import create_l2_loss
from dip_core.utils.metrics import PSNR
from dip_core.utils.debugging import Debugger
from dip_core.utils.plotting import plot_network, plot_tensor

from dip_core.networks.factory import DefaultNetworkFactory
from dip_core.training.default_training_step import DefaultTrainingStep
from dip_core.training.default_trainer import DefaultTrainer

from dip_core.base_implementations.default_operator import DefaultOperator
from dip_core.base_implementations.default_stopper import DefaultStopper
from dip_core.base_implementations.default_optimizer import DefaultOptimizerFactory

from dip_core.base_implementations.config_manager import ConfigManager
from dip_core.base_implementations.run_folder_manager import RunFolderManager

from experiments.denoising.denoising_logger import DenoisingLogger


CURRENT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
EXPERIMENT_NAME = 'denoising'


if __name__ == "__main__":

    # ------------------------------
    # 1. Load + validate config
    # ------------------------------

    template_path = os.path.join(CURRENT_DIR, "config_template.yml")
    user_config_path = os.path.join(CURRENT_DIR, "config.yml")
    log_dir = os.path.join(CURRENT_DIR, f"logs/{EXPERIMENT_NAME}")

    cfg_manager = ConfigManager(template_path)
    config = cfg_manager.load(user_config_path)

    cfg_manager.validate(config)

    # ------------------------------
    # 2. Prepare run folder
    # ------------------------------
    folder_mgr = RunFolderManager(exp_type=EXPERIMENT_NAME, log_dir=LOG_DIR)
    run_dir = folder_mgr.create_run_folder()
    folder_mgr.save_config(config, run_dir)

    # ------------------------------
    # 3. Set random seeds
    # ------------------------------
    set_seed(config["seed"])

    # ------------------------------
    # 4. Load data
    # ------------------------------
    gt = load_image(config["ground_truth_image"])
    noise = torch.randn_like(gt) * config["sigma"]/255.0
    noisy_image = gt + noise
    z = torch.randn(config["noise_shape"]) * config["std_inp_noise"]

    # plot_tensor(noisy_image)
    # plot_tensor(gt)
    # plot_tensor(z)

    # ------------------------------
    # 5. Build network
    # ------------------------------
    model = DefaultNetworkFactory().create_network(
        config["network"],
        config=config["description"]
    )

    # ------------------------------
    # 6. Move to GPU if needed
    # ------------------------------
    if config["use_gpu"] and torch.cuda.is_available():
        model = model.cuda()
        gt = gt.cuda()
        noisy_image = noisy_image.cuda()
        noise = noise.cuda()
        z = z.cuda()

    # ------------------------------
    # 7. Build training components
    # ------------------------------
    loss_fn = create_l2_loss()
    operator = DefaultOperator()
    stopper = DefaultStopper(config["iterations"])
    optimizer = DefaultOptimizerFactory(config["lr"]).create_optimizer(model.parameters())
    step_fn = DefaultTrainingStep()

    metrics = [PSNR()]
    logger = DenoisingLogger(gt, metrics_list=metrics, log_dir=run_dir)

    logger.save_config(config)
    logger.gt_metrics(noisy_image)

    # ------------------------------
    # 8. Run training
    # ------------------------------
    trainer = DefaultTrainer(
        model=model,
        training_step=step_fn,
        operator=operator,
        stopper=stopper,
        optimizer=optimizer,
        loss_fn=loss_fn,
        noise=noise,
        target=noisy_image,
        logger=logger,
        config=config,
        image_per=config["image_per"]
    )

    # ------------------------------
    # 8. Run debuggings 
    # ------------------------------    

    # debugger = Debugger(model)
    # debugger.run("memorization", gt)

    # print(debugger.get_log())

    # plot_network(model)



    trainer.train()
