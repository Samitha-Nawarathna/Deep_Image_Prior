import yaml
import os
import matplotlib.pyplot as plt

from dip_core.networks.factory import *

from dip_core.base_implementations.default_optimizer import DefaultOptimizerFactory
from dip_core.base_implementations.default_logger import * 

from dip_core.utils.seeds import *
from dip_core.utils.image_io import load_image
from dip_core.utils.plotting import plot_tensor 

from dip_core.utils.losses import create_l2_loss

from dip_core.base_implementations.default_operator import DefaultOperator
from dip_core.base_implementations.default_stopper import DefaultStopper

from dip_core.training.default_trainer import DefaultTrainer
from dip_core.training.default_training_step import DefaultTrainingStep

from experiments.denoising.denoising_logger import DenoisingLogger

from dip_core.utils.metrics import PSNR



CURRENT_DIR = os.path.dirname(__file__)

if __name__ == '__main__':
    #load config
    config = yaml.safe_load(open(os.path.join(CURRENT_DIR, 'config.yml'), 'r'))

    #setup experiment


    #set random seed
    set_seed(config['seed'])

    #load image
    gt = load_image(config['ground_truth_image'])

    noise = torch.randn_like(gt) * config['sigma']
    noisy_image = gt + noise

    z = torch.randn(config['noise_shape']) * config['sigma']

    #metrics list
    metrics_list = [PSNR()]

    #set up logger
    logger = DenoisingLogger(gt,metrics_list=metrics_list, log_dir=config['log_dir'])
    logger.save_config(config)


    ## plot tensor images
    plot_tensor(gt)
    plot_tensor(noisy_image)
    plot_tensor(z)

    

    #set up network
    network_factory = DefaultNetworkFactory()
    model = network_factory.create_network(config['network'], config=config['description'])

    # #plot network outputs before training
    # plot_tensor(model(z))
    # plot_tensor(model(noisy_image))

    #setup other requirmed modules for training
    loss_fn = create_l2_loss()
    operator = DefaultOperator()
    stopper = DefaultStopper(config['iterations'])
    optimizer = DefaultOptimizerFactory(config['lr']).create_optimizer(model.parameters())

    step_fn = DefaultTrainingStep()

    if config['use_gpu']:
        model = model.cuda()
        noisy_image = noisy_image.cuda()
        z = z.cuda()
        noise = noise.cuda()

    trainer = DefaultTrainer(
        model, step_fn, operator, stopper, optimizer, loss_fn, noise, noisy_image, logger, config, image_per=500
    )

    trainer.train()





    

