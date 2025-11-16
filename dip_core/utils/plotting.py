
import matplotlib.pyplot as plt

def plot_curve(values, save_path):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(save_path)
    plt.close()
    