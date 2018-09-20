# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 05:15:06 2018

@author: Iwan




i = itemp(:,:,1);
rtemp = min(i);         % find the min. value of pixels in all the columns (row vector)
rmin = min(rtemp);      % find the min. value of pixel in the image
rtemp = max(i);         % find the max. value of pixels in all the columns (row vector)
rmax = max(rtemp);      % find the max. value of pixel in the image
m = 255/(rmax - rmin);  % find the slope of line joining point (0,255) to (rmin,rmax)
c = 255 - m*rmax;       % find the intercept of the straight line with the axis
i_new = m*i + c;        % transform the image according to new slope
figure,imshow(i);       % display original image
figure,imshow(i_new);   % display transformed image

"""
import matplotlib.pyplot as plt
import cv2

def histogram_plot(img,title):
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([img],[channel],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(title)
    plt.show()
    
def contrast_stretch(image):
    h=image.shape[0]
    w=image.shape[1]
    minI=image[0,0]
    maxI=image[0,0]
    minO=0
    maxO=255
    
    #max dan min
    for y in range(0,h):
        for x in range(0,w):
            if image[y,x]>maxI:
                maxI=image[y,x]
            if image[y,x]<minI:
                minI=image[y,x]
                
    #change pixel values
    for y in range(0,h):
        for x in range(0,w):
            image[y,x]= (image[y,x]-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return image

if __name__ == '__main__':
    filename='test.jpg'
    image_load = cv2.imread(filename)
    cv2.imshow('original picture',image_load)
    
    image=image_load.copy()
    print('This image is:', type(image),' with dimensions:', image.shape)
    red_channel=image[:,:,0]
    green_channel=image[:,:,1]
    blue_channel=image[:,:,2]
    
    histogram_plot(image_load,'original image')
    
    new_red=contrast_stretch(red_channel)
    new_blue=contrast_stretch(blue_channel)
    new_green=contrast_stretch(green_channel)
    new_image=cv2.merge((new_blue,new_red,new_green))
    
    cv2.imwrite('contrast_stretch_picture.png',new_image)
    cv2.imshow('constrast stretch picture',new_image)
    histogram_plot(new_image,'constrast stretch image')
    
    while True:
        k = cv2.waitKey(0) & 0xFF     
        if k == 27: break             # ESC key to exit 
    cv2.destroyAllWindows()

    


