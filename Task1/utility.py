
import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d

def PaddingZeroes(source: np.ndarray, f: int) -> np.ndarray:
   
    img = np.copy(source)

    # Calculate Padding size
    p = int((f - 1) / 2)

    # Apply Zero Padding
    After_Image = np.pad(img, (p, p), 'constant', constant_values=0)

    if len(img.shape) == 3:
        return After_Image[:, :, p:-p]
    elif len(img.shape) == 2:
        return After_Image


def Kernel(size: int, mode: str, sigma: [int, float] = None) -> np.ndarray:
    
    if mode == 'ones':
        return np.ones((size, size))
    elif mode == 'gaussian':
        space = np.linspace(np.sqrt(sigma), -np.sqrt(sigma), size * size)
        kernel1d = np.diff(st.norm.cdf(space))
        kernel2d = np.outer(kernel1d, kernel1d)
        return kernel2d / kernel2d.sum()


def Convolve(source: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
   
    img = np.copy(source)

    # Check for Grayscale Image
    if len(img.shape) == 2 or img.shape[-1] == 1:
        conv = convolve2d(img, kernel, mode)
        return conv.astype('uint8')

    After_Image = []
    # Apply Kernel using Convolution
    for channel in range(img.shape[-1]):
        conv = convolve2d(img[:, :, channel], kernel, mode)
        After_Image.append(conv)
    return np.stack(After_Image, -1)
