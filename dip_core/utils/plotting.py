
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch
from torchviz import make_dot

def plot_tensor(tensor):
    #validate tensor can be turn into an image
    

    plt.figure()
    image =  (
        tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)

    plt.imshow(image)
    plt.show()

def plot_curve(values, save_path):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(save_path)
    plt.close()

def plot_network(model, dims = (1, 3, 512, 512)):
    try:
        dummy = torch.randn(*dims)
        out = model(dummy)

        make_dot(out, params=dict(model.named_parameters())).render("network_graph", format="png")
    except:
        print("Could not plot network graph. inline plot: \n")
        summary(model, input_size=dims[1:])
    