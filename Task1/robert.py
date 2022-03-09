import cv2
from convolution import convolve
import numpy as np

roberts_cross_v = np.array([[-1, 0],
                            [0, 1]])

roberts_cross_h = np.array([[0, -1],
                            [1, 0]])

def Robert_Mask(img: np.array) -> np.array:
    vertical = convolve(img, roberts_cross_v)
    horizontal = convolve(img, roberts_cross_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    cv2.imwrite("OUT_IMG/Robert.jpg", edged_img) #when try to preview image direct it corrupted so i have to save it first then preview

