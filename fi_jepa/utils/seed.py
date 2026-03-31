from __future__ import annotations
import matplotlib.pyplot as plt


def plot_training_curve(losses):

    plt.figure()

    plt.plot(losses)

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Curve")

    plt.grid(True)

    plt.show()