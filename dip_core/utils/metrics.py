
import torch
import torch.nn.functional as F

def mse(x, y):
    return F.mse_loss(x, y)

def psnr(x, y, max_val=1.0):
    mse_val = mse(x, y).item()
    return 20 * torch.log10(max_val / (mse_val ** 0.5 + 1e-8))
    