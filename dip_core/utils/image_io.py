
from PIL import Image
import numpy as np
import torch

def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0)

def save_image(tensor, path):
    arr = (
        tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    Image.fromarray(arr).save(path)
    