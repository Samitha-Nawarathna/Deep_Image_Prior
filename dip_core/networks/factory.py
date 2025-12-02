import torch
import torch.nn as nn
import torch.nn.functional as F
from dip_core.abstractions.network_factory import NetworkFactory
from dip_core.networks.hourglass import HourglassNetwork

class DefaultNetworkFactory(NetworkFactory):
    def __init__(self):
        self._registry = {
            "hourglass": HourglassNetwork
            # add other network types here
        }

    def create_network(self, network_type: str, config: dict):
        network_type = network_type.lower()
        if network_type not in self._registry:
            raise ValueError(f"Unknown network type: {network_type}")

        net_cls = self._registry[network_type]

        return net_cls(**config)