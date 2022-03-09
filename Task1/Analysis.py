import cv2
import numpy as np
import matplotlib.pyplot as plt
def Histogram(img: np.array):
    b, g, r = cv2.split(img)  #split img to RGB component
    k=0
    pixel_count=np.zeros((256),dtype=int) #array from 0 to 256
    while k<256:
        pixel_count[k]=np.count_nonzero(r==k) #count Red component
        k+=1
    pixel_value=np.arange(0,256,1)
    plt.bar(pixel_value, pixel_count,color="red")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Histogram')

    m = 0
    pixel_count = np.zeros((256), dtype=int)
    while m < 256:
        pixel_count[m] = np.count_nonzero(g == m) #count green component
        m += 1
    pixel_value = np.arange(0, 256, 1)
    plt.bar(pixel_value, pixel_count,color="green")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Histogram')

    n = 0
    pixel_count = np.zeros((256), dtype=int)
    while n < 256:
        pixel_count[n] = np.count_nonzero(b == n)#count Blue component
        n += 1
    pixel_value = np.arange(0, 256, 1)
    plt.bar(pixel_value, pixel_count,color="Blue")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Histogram')

    plt.show()

def Distribution(img: np.array):
    (r, g, b) = cv2.split(img)
    k=0
    pixel_count=np.zeros((256),dtype=int)
    while k<256:
        pixel_count[k]=np.count_nonzero(r==k)
        k+=1
    pixel_value=np.arange(0,256,1)
    plt.plot(pixel_value, pixel_count,color="red")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Distribution ')


    m = 0
    pixel_count = np.zeros((256), dtype=int)
    while m < 256:
        pixel_count[m] = np.count_nonzero(g == m)
        m += 1
    pixel_value = np.arange(0, 256, 1)
    plt.plot(pixel_value, pixel_count,color="green")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Distribution ')

    n= 0
    pixel_count = np.zeros((256), dtype=int)
    while n < 256:
        pixel_count[n] = np.count_nonzero(b == n)
        n += 1
    pixel_value = np.arange(0, 256, 1)
    plt.plot(pixel_value, pixel_count,color="blue")
    plt.ylabel('pixel_count')
    plt.xlabel('pixel_value')
    plt.title('Distribution ')

    plt.show()
