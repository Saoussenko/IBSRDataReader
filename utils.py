import numpy as np


def perform_rotation(img):
    img = img.reshape([256, 256])
    return np.flipud(img)
