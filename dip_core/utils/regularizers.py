import torch

class TVRegularizer:
    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tv_loss = torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + \
                  torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))
        return self.weight * tv_loss