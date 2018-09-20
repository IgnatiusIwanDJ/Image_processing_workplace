# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:32:29 2018

@author: Iwan
"""

import cv2
import matplotlib.pyplot as plt

def histogram_plot(img,title):
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([img],[channel],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(title)
    plt.show()

def round_number(x):
    n = int(x)
    return n if n-1 < x <= n else n+1

def histeq(img,width,height,no_bin):
    pixel_name=[]
    pixel_occurance=[]
    no_pixel=width*height
    
    #index pixel
    for row in range(width):
        for column in range(height):
            if img[column,row] not in pixel_name:
                pixel_name.append(img[column,row])
                
    #sort index
    pixel_name.sort()
    
    #count occurence
    for pixel in pixel_name:
        count=0
        for row in range(width):
            for column in range(height):
                if img[column,row]==pixel:
                    count+=1
        pixel_occurance.append(count)
    
    #find the cumulative histogram
    cum_hist=[]
    cumulative=0
    for pixel_value in pixel_occurance:
        cumulative+=pixel_value
        cum_hist.append(cumulative)
        
    #cumulative prob
    cum_prob=[]
    for pixel in cum_hist:
        prob=float(pixel/no_pixel)
        cum_prob.append(prob)
    
    #calculate final value
    list_of_final=[]
    for pixel in cum_prob:
        fin_pixel=pixel*no_bin
        fin_pixel=round_number(fin_pixel)
        list_of_final.append(fin_pixel)
        
    #make dictionary
    dictionary=dict(zip(pixel_name,list_of_final))
    
    #construct image
    for row in range(width):
            for column in range(height):
                img[column,row]=dictionary[img[column,row]]
                
    return img

#filename='gambar_jpg_1.jpg'
filename='test.jpg'
img = cv2.imread(filename,-1)
image=img.copy()
print('This image is:', type(img),' with dimensions:', img.shape)
img_shape = img.shape
height = img_shape[0]
width = img_shape[1]

histogram_plot(img,'original image')
cv2.imshow('our original picture',img)

#histeq
new_image_red=histeq(image[:,:,0],width,height,256)
new_image_green=histeq(image[:,:,1],width,height,256)
new_image_blue=histeq(image[:,:,2],width,height,256)
new_image=cv2.merge((new_image_blue,new_image_red,new_image_green))

histogram_plot(new_image,'histogram equalization image')
cv2.imshow('final',new_image)
cv2.imwrite('histogram_picture.png',new_image)

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()