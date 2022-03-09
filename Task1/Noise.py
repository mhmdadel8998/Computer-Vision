#
# Noise libs Implementation
#

# Imports
import numpy as np


def Resize(source: np.ndarray, high: [int, float], low: [int, float]) -> np.ndarray:
    
    After_Image = np.copy(source)
    After_Image[After_Image > high] = high
    After_Image[After_Image < low] = low
    return After_Image


def uniform_noise(source: np.ndarray, snr: float) -> np.ndarray:
   
    # Create Noise Mask, Following a Uniform Distribution
    noise = np.random.uniform(-255, 255, size=source.shape)
    img = np.copy(source)

    # Apply Uniform Noise Mask
    After_Image = img * snr + noise * (1 - snr)

    # Resizeping Image in range 0, 255
    After_Image = Resize(After_Image, 255, 0)
    return After_Image.astype(int)


def gaussian_noise(source: np.ndarray, sigma: [int, float], snr: float) -> np.ndarray:
   
    # Create Noise Mask, Following the Gaussian Distribution
    noise = np.random.normal(0, sigma, size=source.shape)
    img = np.copy(source)
    After_Image = img * snr + noise * (1 - snr)

    # resizing Image in range 0, 255
    After_Image = Resize(After_Image, 255, 0)
    return After_Image.astype(int)


def salt_pepper_noise(source: np.ndarray, snr: float) -> np.ndarray:
    
    # Create Noise Mask, Randomly select pixels to be either 0 or 255
    noise = np.random.choice((0, 1, 2), size=source.shape, p=[snr, (1 - snr) / 2, (1 - snr) / 2])
    img = np.copy(source)

    # Apply Noise Mask
    img[noise == 1] = 255
    img[noise == 2] = 0
    return img
