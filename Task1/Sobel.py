import numpy as np
from convolution import convolve
X_kernel = (np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]))

Y_kernel = (np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]]))
XY_kernel = (np.array([[-2,-2, 0],
                    [-2, 0, 2],
                    [0, 0, 2]]))

outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]])
def Sobel_X(img: np.array) -> np.array:
    imgx = convolve(img, X_kernel)
    return imgx
def Sobel_Y(img: np.array) -> np.array:
    imgy = convolve(img, Y_kernel)
    return imgy
def Sobel_XY(img: np.array) -> np.array:
    imgx = convolve(img, X_kernel)
    imgxy = convolve(imgx, Y_kernel)
    return imgxy
