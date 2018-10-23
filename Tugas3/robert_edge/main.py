# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:40:15 2018

@author: Iwan
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('test_image.png',cv2.IMREAD_GRAYSCALE)

gray_img = img.copy()

h, w = gray_img.shape

#2x2 robert kernel
horizontal = np.array([[1, 0], 
                       [0,-1]])  
vertical = np.array([[0,1], 
                     [-1,0]])  

newgradientImage = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) 

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) 

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag


plt.figure()
plt.title('robert edge.png')
plt.imsave('test_robert_2.png', newgradientImage, cmap='gray', format='png')
plt.imshow(newgradientImage, cmap='gray')
plt.show()
