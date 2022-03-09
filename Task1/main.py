

import numpy as np
import cv2
from Noise import *
from normal_filters import *
from utility import *
from robert import *
from Sobel import*
from Analysis import *
from Canny import *
from Prewitt import *
import equalization 
import normalize


image = cv2.imread("2.jpg")
image = cv2.resize(image, (400,400))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.convertScaleAbs(image_gray, alpha=1.10, beta=-20)

  

B_img= cv2.GaussianBlur(image_gray, (5,5), 0)

sx=Sobel_X(B_img)
sy=Sobel_Y(B_img)
sxy=Sobel_XY(B_img)
can=Canny_Mask(B_img)
pre=Prewitt_Mask(image_gray)
Edged=Robert_Mask(image_gray)


cv2.imshow("Gray Scaled Image",image_gray)
cv2.waitKey(0)

cv2.imshow("Blurred Image",B_img)
cv2.waitKey(0)

cv2.imshow("sobelx",sx)
cv2.waitKey(0)

cv2.imshow("sobely",sy)
cv2.waitKey(0)

cv2.imshow("sobelxy",sxy)
cv2.waitKey(0)

cv2.imshow("Canny",can)
cv2.waitKey(0)

Eimage= cv2.imread('OUT_IMG/Robert.jpg',flags=0)
cv2.imshow("Robert_Image",Eimage)
cv2.waitKey(0)

Pimage= cv2.imread("OUT_IMG/prewitt_image.jpg")
cv2.imshow("Prewitt Image",Pimage)
cv2.waitKey(0)

his=Histogram(N_img)
dis=Distribution(N_img)

equalized_image = equalization.equalization(image_gray)
cv2.imshow('Equalization',equalized_image)
cv2.waitKey(0)

cv2.imshow('Normalization',normalize.Normalization(image_gray))
cv2.waitKey(0)



uniform_noise_img = uniform_noise(image_gray, 0.9)
cv2.imshow('Uniform noise',uniform_noise_img.astype(np.uint8))
cv2.waitKey(0)


filtered_img = average_filter(uniform_noise_img,3)
cv2.imshow("Average Filter with uniform noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()

filtered_img = median_filter(uniform_noise_img,3,1)
cv2.imshow("Median Filter with uniform noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


filtered_img = gaussian_filter(uniform_noise_img,3)
cv2.imshow("Gaussian Filter with uniform noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


salt_pepper_noise_img = salt_pepper_noise(image_gray, 0.9)
cv2.imshow('Salt and Pepper noise',salt_pepper_noise_img.astype(np.uint8))
cv2.waitKey(0)


filtered_img = average_filter(salt_pepper_noise_img,3)
cv2.imshow("Average Filter with salt and pepper noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()

filtered_img = median_filter(salt_pepper_noise_img,3,1)
cv2.imshow("Median Filter with salt and pepper noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


filtered_img = gaussian_filter(salt_pepper_noise_img,3)
cv2.imshow("Gaussian Filter with salt and pepper noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


gaussian_noise_img = gaussian_noise(image_gray ,64,0.9)
cv2.imshow('Gaussian noise',gaussian_noise_img.astype(np.uint8))
cv2.waitKey(0)


filtered_img = average_filter(gaussian_noise_img,3)
cv2.imshow("Average Filter with gaussian noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()

filtered_img = median_filter(gaussian_noise_img,3,1)
cv2.imshow("Median Filter with gaussian noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


filtered_img = gaussian_filter(gaussian_noise_img,3)
cv2.imshow("Gaussian Filter with gaussian noise.jpg",filtered_img.astype(np.uint8))
cv2.waitKey()


 







