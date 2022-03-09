import numpy as np
from Sobel import X_kernel,Y_kernel
from convolution import convolve
import cv2
#Define gradient magnitude function
def gradient_mag(fx, fy) :
    grad_mag = np.hypot(fx, fy)
    return grad_mag/np.max(grad_mag)

#Find closest direction D*
def closest_dir_function(grad_dir):
    closest_dir_arr = np.zeros(grad_dir.shape)
    for i in range(1, int(grad_dir.shape[0] - 1)):
        for j in range(1, int(grad_dir.shape[1] - 1)):

            if ((grad_dir[i, j] > -22.5 and grad_dir[i, j] <= 22.5) or (
                    grad_dir[i, j] <= -157.5 and grad_dir[i, j] > 157.5)):
                closest_dir_arr[i, j] = 0

            elif ((grad_dir[i, j] > 22.5 and grad_dir[i, j] <= 67.5) or (
                    grad_dir[i, j] <= -112.5 and grad_dir[i, j] > -157.5)):
                closest_dir_arr[i, j] = 45

            elif ((grad_dir[i, j] > 67.5 and grad_dir[i, j] <= 112.5) or (
                    grad_dir[i, j] <= -67.5 and grad_dir[i, j] > -112.5)):
                closest_dir_arr[i, j] = 90

            else:
                closest_dir_arr[i, j] = 135

    return closest_dir_arr

#Convert to thinned edge
def non_maximal_suppressor(grad_mag, closest_dir):
    thinned_output = np.zeros(grad_mag.shape)
    for i in range(1, int(grad_mag.shape[0] - 1)):
        for j in range(1, int(grad_mag.shape[1] - 1)):

            if (closest_dir[i, j] == 0):
                if ((grad_mag[i, j] > grad_mag[i, j + 1]) and (grad_mag[i, j] > grad_mag[i, j - 1])):
                    thinned_output[i, j] = grad_mag[i, j]
                else:
                    thinned_output[i, j] = 0

            elif (closest_dir[i, j] == 45):
                if ((grad_mag[i, j] > grad_mag[i + 1, j + 1]) and (grad_mag[i, j] > grad_mag[i - 1, j - 1])):
                    thinned_output[i, j] = grad_mag[i, j]
                else:
                    thinned_output[i, j] = 0

            elif (closest_dir[i, j] == 90):
                if ((grad_mag[i, j] > grad_mag[i + 1, j]) and (grad_mag[i, j] > grad_mag[i - 1, j])):
                    thinned_output[i, j] = grad_mag[i, j]
                else:
                    thinned_output[i, j] = 0

            else:
                if ((grad_mag[i, j] > grad_mag[i + 1, j - 1]) and (grad_mag[i, j] > grad_mag[i - 1, j + 1])):
                    thinned_output[i, j] = grad_mag[i, j]
                else:
                    thinned_output[i, j] = 0

    return thinned_output / np.max(thinned_output)
#Function to include weak pixels that are connected to chain of strong pixels
def DFS(img) :
    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            if(img[i, j] == 1) :
                t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                            img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                if(t_max == 2) :
                    img[i, j] = 2
#Hysteresis Thresholding
def hysteresis_thresholding(img):
    low_ratio = 0.10
    high_ratio = 0.30
    diff = np.max(img) - np.min(img)
    t_low = np.min(img) + low_ratio * diff
    t_high = np.min(img) + high_ratio * diff

    temp_img = np.copy(img)

    # Assign values to pixels
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            # Strong pixels
            if (img[i, j] > t_high):
                temp_img[i, j] = 2
            # Weak pixels
            elif (img[i, j] < t_low):
                temp_img[i, j] = 0
            # Intermediate pixels
            else:
                temp_img[i, j] = 1

    # Include weak pixels that are connected to chain of strong pixels
    total_strong = np.sum(temp_img == 2)
    while (1):
        DFS(temp_img)
        if (total_strong == np.sum(temp_img == 2)):
            break
        total_strong = np.sum(temp_img == 2)

    # Remove weak pixels
    for i in range(1, int(temp_img.shape[0] - 1)):
        for j in range(1, int(temp_img.shape[1] - 1)):
            if (temp_img[i, j] == 1):
                temp_img[i, j] = 0

    temp_img = temp_img / np.max(temp_img)
    return temp_img
N_img= cv2.imread('1.jpg')
G_img=cv2.cvtColor(N_img, cv2.COLOR_BGR2GRAY) #flage=0 for gray scale
B_img= cv2.GaussianBlur(G_img, (3,3), 0)
def Canny_Mask(img: np.array):
    #Find gradient Fx
    x_grad = convolve(img,X_kernel)
    # Find gradient Fy
    y_grad = convolve(img,Y_kernel)
    # Compute edge strength
    grad_mag = gradient_mag(x_grad, y_grad)
    # Compute direction of gradient
    grad_dir = np.degrees(np.arctan2(y_grad, x_grad))
    # Phase 2 : Non maximal suppression
    closest_dir = closest_dir_function(grad_dir)
    thinned_output = non_maximal_suppressor(grad_mag, closest_dir)
    # # Phase 3 : Hysteresis Thresholding
    output_img = hysteresis_thresholding(thinned_output)
    return output_img
