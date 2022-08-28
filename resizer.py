from skimage import transform
import numpy as np
import random
import cv2
import os
from time import sleep
from tqdm import tqdm

def resize_image(image, min_dim=None, max_dim=None, mode="square"):

    h, w = image.shape[:2]
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale < 1:
        image = transform.resize(image, (round(h * scale), round(w * scale)), preserve_range=True)
    if scale < 1:
        pass
    if scale == 1:
        pass
    
    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)    
    return image

folder_path = 'C:/Users/LENOVO/Downloads/2400_dataset/dataset'
target_path = 'C:/Users/LENOVO/Downloads/2400_dataset/resized'

for i, filename in zip(tqdm(range(len(os.listdir(folder_path))), desc= "Converting..."), os.listdir(folder_path)):
    f = os.path.join(folder_path, filename)
    image = cv2.imread(f)
    new_image = resize_image(image, min_dim=512, max_dim=512, mode="square")
    cv2.imwrite(target_path + "/" + filename, new_image)