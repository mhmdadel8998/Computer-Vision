import cv2
from convolution import convolve
import numpy as np

Hx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
Hy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])

def Prewitt_Mask(img: np.array):

    pre_x = convolve(img, Hx)
    pre_y = convolve(img, Hy)
    #calculate the gradient magnitude of vectors
    pre_out = np.sqrt(np.power(pre_x, 2) + np.power(pre_y, 2))
    # mapping values from 0 to 255
    pre_out = (pre_out / np.max(pre_out)) * 255
    cv2.imwrite('OUT_IMG/prewitt_image.jpg', pre_out)
#when try to preview image direct it corrupted so i have to save it first then preview