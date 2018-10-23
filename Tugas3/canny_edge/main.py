# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 07:23:59 2018

@author: Iwan

1.Apply Gaussian filter to smooth the image in order to remove the noise
2.Find the intensity gradients of the image
3.Apply non-maximum suppression to get rid of spurious response to edge detection
4.Apply double threshold to determine potential edges
5.Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

Credit to : machinelearninggod (Stefan Stavrev) for hysteresis function
"""
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

def convolve_3x3(image, kernel):
    h,w=image.shape
    newgradientImage = np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            output= (kernel[0, 0] * image[i - 1, j - 1]) +\
                    (kernel[0, 1] * image[i - 1, j]) + \
                    (kernel[0, 2] * image[i - 1, j + 1]) + \
                    (kernel[1, 0] * image[i, j - 1]) + \
                    (kernel[1, 1] * image[i, j]) + \
                    (kernel[1, 2] * image[i, j + 1]) + \
                    (kernel[2, 0] * image[i + 1, j - 1]) + \
                    (kernel[2, 1] * image[i + 1, j]) + \
                    (kernel[2, 2] * image[i + 1, j + 1])
            newgradientImage[i - 1, j - 1] = output
    
    return newgradientImage

def convolve_kernel_edge(image, kernel_x,kernel_y):
    h,w=image.shape
    newgradientImage = np.zeros((h, w))
    directions = np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad= (kernel_x[0, 0] * image[i - 1, j - 1]) +\
                    (kernel_x[0, 1] * image[i - 1, j]) + \
                    (kernel_x[0, 2] * image[i - 1, j + 1]) + \
                    (kernel_x[1, 0] * image[i, j - 1]) + \
                    (kernel_x[1, 1] * image[i, j]) + \
                    (kernel_x[1, 2] * image[i, j + 1]) + \
                    (kernel_x[2, 0] * image[i + 1, j - 1]) + \
                    (kernel_x[2, 1] * image[i + 1, j]) + \
                    (kernel_x[2, 2] * image[i + 1, j + 1])
                    
            verticalGrad= (kernel_y[0, 0] * image[i - 1, j - 1]) +\
                    (kernel_y[0, 1] * image[i - 1, j]) + \
                    (kernel_y[0, 2] * image[i - 1, j + 1]) + \
                    (kernel_y[1, 0] * image[i, j - 1]) + \
                    (kernel_y[1, 1] * image[i, j]) + \
                    (kernel_y[1, 2] * image[i, j + 1]) + \
                    (kernel_y[2, 0] * image[i + 1, j - 1]) + \
                    (kernel_y[2, 1] * image[i + 1, j]) + \
                    (kernel_y[2, 2] * image[i + 1, j + 1])
                    
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag
            dir_i=float(math.fmod(math.atan2(verticalGrad,horizontalGrad)+math.pi,math.pi)/math.pi)*8.0
            directions[i-1,j-1]=dir_i
            
    return newgradientImage,directions

def GaussKernel(size,sigma):
    gauss=np.zeros((size,size))
    m=size//2
    for x in range(-m,m+1):
        for y in range(-m,m+1):
            x1=2*np.pi*sigma**2
            x2=np.exp(-(x**2+y**2)/(2*sigma**2))
            gauss[x+m,y+m]=(1/x1)*x2
    return gauss

def trace_and_threshold(E_nms, E_bin, i, j, t_low):
    E_bin[i,j] = 255
    
    jL = np.max([j-1, 0])
    jR = np.min([j+1, E_bin.shape[1]])
    
    iT = np.max([i-1, 0])
    iB = np.min([i+1, E_bin.shape[0]])
    
    for ii in np.arange(iT, iB):
        for jj in np.arange(jL, jR):
            if E_nms[ii,jj] >= t_low and E_bin[ii,jj] == 0:
                trace_and_threshold(E_nms, E_bin, ii, jj, t_low)
                
    return
    
if __name__=='__main__':
    img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
    image=img.copy()
    
    print('This image is:', type(image),' with dimensions:', image.shape)
    h,w=image.shape
    
    # blur gauss
    gauss_kernel=GaussKernel(3,1.0)
    gauss_image=convolve_3x3(image,gauss_kernel)
    
    #prewit magnitude
    Gx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])

    Gy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])
    
    mag_image,directions=convolve_kernel_edge(gauss_image,Gx,Gy)
    
    #non-maximum suppression
    non_max_img=np.zeros((h, w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            
            if directions[i,j] <=1 or directions[i,j] > 7:
                if(mag_image[i,j] > mag_image[i,j-1] and mag_image[i,j] > mag_image[i,j+1]):
                    non_max_img[i,j]=mag_image[i,j]
            elif directions[i,j] >1 or directions[i,j] <= 3:
                if(mag_image[i,j] > mag_image[i-1,j+1] and mag_image[i,j] > mag_image[i+1,j-1]):
                    non_max_img[i,j]=mag_image[i,j]
            elif directions[i,j] >3 or directions[i,j] <= 5:
                if (mag_image[i,j] > mag_image[i-1,j] and mag_image[i,j] > mag_image[i+1,j]):
                    non_max_img[i,j]=mag_image[i,j]
            elif directions[i,j] >5 or directions[i,j] <= 7:
                if(mag_image[i,j] > mag_image[i-1,j-1] and mag_image[i,j] > mag_image[i+1,j+1]):
                    non_max_img[i,j]=mag_image[i,j]
            else:
                non_max_img[i,j]=0
    
    #set low and high threshold
    non_max_img=(non_max_img/np.max(non_max_img))*255
    high_t=15
    low_t=4
    print(high_t)
    print(low_t)
    #print(non_max_img)
    
    #threshold and hystheris
    new_bin=np.zeros((h, w))
    for i in np.arange(1, h-1):
        for j in np.arange(1, w-1):
            if non_max_img[i,j] >= high_t and new_bin[i,j] == 0:
                trace_and_threshold(non_max_img, new_bin, i, j, low_t)

    plt.figure()
    plt.title('output_image.png')
    plt.imsave('canny_edge_2.png', new_bin, cmap='gray', format='png')
    plt.imshow(new_bin, cmap='gray')
    plt.show()
   
