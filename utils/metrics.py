'''
metrics.py

682 - Neural Networks - Project

Sreerag Iyer, Chirag Trasikar

This file contains some functions to generate
evaluation metrics for our trained models.
'''
import numpy as np

# psnr: Calculates the peak signal to noise ratio for a set of interpolated images and their ground truth images 
def psnr(gt_images, sr_images):
    mse_loss = np.mean((gt_images - sr_images) ** 2)
    psnr_loss = 10 * np.log10((255. ** 2) / mse_loss)

    return psnr_loss