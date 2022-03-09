#
# Low Pass libs Implementations
#

import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d
from utility import *



def average_filter(source: np.ndarray, shape: int = 3, sigma: float = None) -> np.ndarray:
    
     
    img = np.copy(source)

    # Create the Average Kernel
    kernel = Kernel(shape, 'ones') * (1 / shape ** 2)

    # Check for Grayscale Image
    After_Image = Convolve(img, kernel, 'same')
    return After_Image.astype('uint8')


def gaussian_filter(source: np.ndarray, shape: int = 5, sigma: [int, float] = 64) -> np.ndarray:
   
    img = np.copy(source)

    # Create a Gaussian Kernel
    kernel = Kernel(shape, 'gaussian', sigma)

    # Apply the Kernel
    After_Image = Convolve(img, kernel, 'same')
    return After_Image.astype('uint8')


def median_filter(source: np.ndarray, shape: int, sigma: float = None) -> np.ndarray:
    
    img = np.copy(source)

    # Check image for right dimensions
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    # Create an Array of the same size as input image
    result = np.zeros(img.shape)

    # Pad the Image to obtain a Same Convolution
    img = PaddingZeroes(img, shape)

    for ix, iy, ic in np.ndindex(img.shape):
        # Looping the Image in the X and Y directions
        # Extracting the Kernel
        # Calculating the Median of the Kernel
        kernel = img[ix: ix + shape, iy: iy + shape, ic]
        if kernel.shape == (shape, shape):
            result[ix, iy, ic] = np.median(kernel).astype('uint8')

    return result
