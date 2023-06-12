# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:53:34 2023

@author: eliav
"""
import numpy as np
import matplotlib.pyplot as plt
import math


# Midtern - Ibud tmuna:
#%%
# Question 1:
"Power Transformation"
Lev = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/Lev_noisysquares.png')



# # Log Transform
'''Our image is in double type and hence we strech the image
by m factor, and then normalize it'''
m=100000
logLev = (np.log(1+m*Lev))/(np.log(1+m))

# Show the original image inside the Improved image:
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(Lev)
# plt.title('Original')
# plt.subplot(1,2,2)
# plt.imshow(logLev)
# plt.title('Log Transformation')
# plt.show()

# Power transformation:
gamma = 0.2
GammaLev = Lev**gamma
plt.figure()
plt.subplot(1,3,1)
plt.imshow(Lev, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(GammaLev,cmap='gray')
plt.title('Gamma=0.5 Transformation')
plt.subplot(1,3,3)
plt.imshow(logLev, cmap='gray')
plt.title('Log transform')
plt.show()

#%%
"Hist Equals"

# Ceil image:
Lev = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/Lev_noisysquares.png')
UInt8Image = np.floor(255*Lev)
int8_image = UInt8Image.astype(np.uint8)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(UInt8Image, cmap='gray')
plt.title("Original image")
plt.show()


plt.subplot(2,2,2)
lHist = plt.hist(x = int8_image.flatten())
plt.title('Original Histogram')
plt.show()

def histeq(img, bins = 256):
    #Takes uint8, returns scaled float
    imgHist = np.histogram(a = img, bins = bins, range = (-1,256))
    N = np.size(img)
    imgNormalizedHist = imgHist[0]/N
    T = np.cumsum(imgNormalizedHist)#Vectorial syntax - every value of the image serves as an index into T
    imgHistEq = T[img]
    return(imgHistEq,T)

LevHisteq,Tnew = histeq(int8_image)

LevHisteqH = np.histogram(a = LevHisteq, bins =  64, range=(0,1))

LevHEHistogram = LevHisteqH[0]
LevHEBins = LevHisteqH[1]

plt.subplot(2,2,3)
plt.imshow(LevHisteq)
plt.title("The image after equal histogram")
plt.subplot(2,2,4)
plt.bar(LevHEBins[:-1],LevHEHistogram,width = 0.9*1/64)
plt.title('Equal Histogram, unnormalized')



#%%
