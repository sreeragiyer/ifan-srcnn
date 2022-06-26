'''
preprocessing.py

682 - Neural Networks - Project

Sreerag Iyer, Chirag Trasikar

This file contains some functions to read and preprocess the data.
'''

import cv2
import os
import numpy as np

# get_chunks: Divides an image into chunks of required shape
def get_chunks(image, chunk_size=(64, 64)):
    # Shapes of chunks and images
    HH, WW = chunk_size     # chunk_height, chuck_width
    H, W, C = image.shape   # image_height, image_width, image_channels 

    # Crop the image so that we can fit chunks perfectly
    H_crop = H - (H % HH)
    W_crop = W - (W % WW)
    image = image[:H_crop, :W_crop, :]

    # Divide the cropped image into chunks
    chunks = []
    rows = np.array_split(image, range(HH, H_crop, HH), axis=0)         # Split vertically
    for row in rows:
        chunks += np.array_split(row, range(WW, W_crop, WW), axis=1)  # Split horizontally

    return chunks


# get_bsds_300_chunks: Loads BDS 300 images as chunks
def get_bsds_300_chunks(path, chunk_size=(64, 64), num_chunks=-1):
    # List image file names
    file_names = os.listdir(path)
    
    # If num_chunks is -1 then we want all possible chunks
    truncate = True
    if num_chunks == -1:
        truncate = False
        
    # Loop over images and gather chunks
    chunks = []
    for i, file_name in enumerate(file_names):
        image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_COLOR)
        
        old_len = len(chunks)
        chunks += get_chunks(image, chunk_size=chunk_size)
        new_len = len(chunks)

        if truncate == True and len(chunks) >= num_chunks:
            break

    # Truncate chunks to required amount
    chunks = np.array(chunks)
    if truncate == True:
        chunks = chunks[:num_chunks]

    return chunks

# get_bsds_300_test_images: Loads BDS 300 test images 
def get_bsds_300_test_images(path):
    # List image file names
    file_names = os.listdir(path)
        
    # Loop over images and gather chunks
    images = []
    for i, file_name in enumerate(file_names):
        image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_COLOR)
        images.append(image)

    return images

# get_dpdd_ifan_test_images: Loads DPDD test images 
def get_dpdd_ifan_test_images(in_path, out_path, scale=0.5, interpolation=cv2.INTER_CUBIC):
    # List image file names
    file_names = os.listdir(in_path)
        
    # Loop over images and gather chunks
    in_images = []
    in_low_res_images = []
    out_images = []
    out_low_res_images = []
    for i, file_name in enumerate(file_names):
        in_image = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_COLOR)
        out_image = cv2.imread(os.path.join(out_path, file_name), cv2.IMREAD_COLOR)

        H, W, C = in_image.shape
        H_out, W_out = H, W
        if H % 2 == 1:
            H_out = H - 1
        
        if W % 2 == 1:
            W_out = W - 1

        
        H_low = int(H_out * scale)
        W_low = int(W_out * scale)

        in_cropped_image = in_image[:H_out, :W_out, :]
        in_images.append(in_cropped_image)
        in_low_res_images.append(cv2.resize(in_cropped_image, dsize=(W_low, H_low), fx=scale, fy=scale, interpolation=interpolation))

        out_cropped_image = out_image[:H_out, :W_out, :]
        out_images.append(out_cropped_image)
        out_low_res_images.append(cv2.resize(out_cropped_image, dsize=(W_low, H_low), fx=scale, fy=scale, interpolation=interpolation))


    return in_images, in_low_res_images, out_images, out_low_res_images


# blur_images: Applies average blur to all images
def blur_images(images, kernel_size=(5, 5), mode='gaussian'):
    blurred_images = []
    for image in images:
        blurred_image = None
        if mode == 'gaussian':
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        else: # if mode == 'mean'
            blurred_image = cv2.blur(image, kernel_size)
        blurred_images.append(blurred_image)

    return np.array(blurred_images)


# resize_images: Converts high-res images to low-res or vice-versa based on the scale and interpolation mode
def resize_images(images, scale=0.5, mode='cubic'):
    _, H, W, _ = images.shape
    H_low = int(H * scale)
    W_low = int(W * scale)

    interpolation = None
    if mode == 'cubic':
        interpolation = cv2.INTER_CUBIC
    else: # mode == 'linear'
        interpolation = cv2.INTER_LINEAR

    low_res_images = []
    for image in images:
        low_res_image = cv2.resize(image, dsize=(H_low, W_low), fx=scale, fy=scale, interpolation=interpolation)
        low_res_images.append(low_res_image)

    return np.array(low_res_images)





