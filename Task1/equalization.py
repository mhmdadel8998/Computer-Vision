import cv2
import numpy as np
import math
import matplotlib.pyplot as plt







# Compute the histogram of the image 
def Histogram(image):
    s =image.shape
    Histo = np.zeros(shape=(256,1))

    for i in range(s[0]):
        for j in range(s[1]):
            k = image[i,j]
            Histo[k,0] = Histo[k,0] + 1
    return Histo

# Histo = Histogram(image_gray) 
# print(Histo)
# plt.plot(Histo)
# plt.show()

# Function that computes equalization and returns the equalized image
def equalization(image):
    s =image.shape
    histo = Histogram(image)
    Nx = histo.reshape(1,256)
    Ny = np.array([])
    Ny=np.append(Ny,Nx[0,0])  
 
    for i in range (255):
        k = Nx[0,i+1]+Ny[i]
        Ny = np.append(Ny,k)

    Ny = np.round(( Ny / (s[0]*s[1]))*(256-1))
    for i in range(s[0]):
        for j in range(s[1]):
            k = image[i,j]
            image[i,j] = Ny[k]
    return image


