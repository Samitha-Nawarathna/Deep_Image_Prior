
import matplotlib.pyplot as plt
import numpy as np

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
    