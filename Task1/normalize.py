import cv2 
from cv2 import waitKey 
import numpy as np 
import math
import matplotlib.pyplot as plt



def Min_Max(image:np.ndarray):
    maximum=0
    minimum=255
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] <= minimum :
                minimum = image[i][j]
            elif image[i][j] >= maximum :
                maximum = image[i][j]
        return minimum,maximum

def Normalization(image:np.ndarray):
    minimum,maximum =Min_Max(image)
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            image[i][j] = (image[i][j]-minimum)*(255/(maximum-minimum))
    return image

