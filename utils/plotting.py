'''
plotting.py

682 - Neural Networks - Project

Sreerag Iyer, Chirag Trasikar

This file contains some functions to generate
plots  metrics for our trained models.
'''
import matplotlib.pyplot as plt
from math import ceil

# plot_images: Plots given images on a grid
def plot_images(images):
    N = len(images)

    cols = ceil(N / 8)
    rows = 8

    H_plot, W_plot = rows * 2, cols * 2
    f = plt.figure()
    f.set_figheight(H_plot)
    f.set_figwidth(W_plot)

    for row in range(rows + 1):
        for col in range(cols + 1):
            image_idx = row * cols + col
            if image_idx < N:
                plt.subplot(rows, cols, image_idx + 1)
                plt.imshow(images[image_idx].astype('uint8'))
            plt.axis('off')
    plt.show()