# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:18:48 2018

@author: Iwan
MATLAB IMPLEMENTATION
function [ lenaOutput] = sobel(X)
%X input color image
X= double(X); height = size(X, 1); width = size(X, 2); channel = size(X, 3);
lenaOutput = X;
Gx = [1 +2 +1; 0 0 0; -1 -2 -1]; Gy = Gx';
for i = 2 : height-1
   for j = 2 : width-1  
       for k = 1 : channel
           tempLena = X(i - 1 : i + 1, j - 1 : j + 1, k);
           a=(sum(Gx.* tempLena));
           x = sum(a);
           b= (sum(Gy.* tempLena));
            y = sum(b);
           pixValue =sqrt(x.^2+ y.^2);
          % pixValue =(x-y);
           lenaOutput(i, j, k) = pixValue;
       end 
   end
end
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test_image.png',cv2.IMREAD_GRAYSCALE)

gray_img = img.copy()

h, w = gray_img.shape

# sobel 3x3 kernel
horizontal = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])  
vertical = np.array([[-1, -2, -1], 
                     [0, 0, 0], 
                     [1, 2, 1]])  

newgradientImage = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag


plt.figure()
plt.title('sobel edge.png')
plt.imsave('sobel_test_2.png', newgradientImage, cmap='gray', format='png')
plt.imshow(newgradientImage, cmap='gray')
plt.show()