
import torch
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as ssim
import numpy as np

from dip_core.abstractions.metric import Metric

class PSNR(Metric):
    def name(self):
        return "PSNR"
    
    def __call__(self, gt, output, **kwargs):
        mse = torch.mean((gt - output) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)  # assuming images in [0,1]
        return psnr.item()

# class SSIM(Metric):
#     def name(self):
#         return "SSIM"
    
#     def __call__(self, gt, output, **kwargs):
#         # placeholder; call actual SSIM function here
        
#         gt_np = gt.detach().cpu().numpy().transpose(1,2,0)
#         output_np = output.detach().cpu().numpy().transpose(1,2,0)
#         ssim_val = ssim(gt_np, output_np, multichannel=True)
#         return ssim_val
    