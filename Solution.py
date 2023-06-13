# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:53:34 2023

@author: eliav
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# Midtern - Ibud tmuna:
#%%
"Question 1:"
#"Power Transformation"
Lev = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/Lev_noisysquares.png')



# # Log Transform
'''Our image is in double type and hence we strech the image
by m factor, and then normalize it'''
m=100000
logLev = (np.log(1+m*Lev))/(np.log(1+m))

# Show the original image inside the Improved image:
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Lev)
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(logLev)
plt.title('Log Transformation')
plt.show()

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
"General hist equals"

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
    imgHist = np.histogram(a = img, bins = bins, range = (-1,256)) # Create histogram plot between -1,256.
    N = np.size(img)
    imgNormalizedHist = imgHist[0]/N
    T = np.cumsum(imgNormalizedHist) #Vectorial syntax - every value of the image serves as an index into T
    imgHistEq = T[img] # Operate C.D.F on the image.
    return(imgHistEq,T)

LevHisteq,Tnew = histeq(int8_image)

LevHisteqH = np.histogram(a = LevHisteq, bins =  64, range=(0,1)) # Create histogram plot between 0,1.

LevHEHistogram = LevHisteqH[0]
LevHEBins = LevHisteqH[1]

plt.subplot(2,2,3)
plt.imshow(LevHisteq)
plt.title("The image after equal histogram")
plt.subplot(2,2,4)
plt.bar(LevHEBins[:-1],LevHEHistogram,width = 0.9*1/64)
plt.title('Equal Histogram, unnormalized')

#%%
"Question 2:"
# a.
import numpy as np
import matplotlib.pyplot as plt

face = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/face.tif')

def centered_correlation(img, mask): # פונקציה המבצעת סינון מרחבי על ידי קורלציה ממורכזת

    height, width = img.shape       # הגדרת גודל התמונה
    k_height, k_width = mask.shape  # הגדרת גודל המסכה וגודל ה"ריפוד" שמורידים
    center_x = (k_width - 1) // 2
    center_y = (k_height - 1) // 2
    
    result = np.zeros((height - k_height + 1, width - k_width + 1), dtype=img.dtype)  # הגדרת מערך בגודל מתאים לתמונה חדשה ללא הריפוד
    
    for i in range(center_y, height - center_y):  # ביצוע הקורלציה בין התמונה למסנן 
        for j in range(center_x, width - center_x):
            roi = img[i - center_y:i + center_y + 1, j - center_x:j + center_x + 1]
            result[i - center_y, j - center_x] = np.sum(roi * mask)
            
    return result

def reverse_mask(mask):             # פונקציה ההופכת את המסכה שמאלה ימינה ולמעלה למטה כדי לאפשר קונבולוציה
    return np.flipud(np.fliplr(mask))

def filter_image(image, mask):      # פונקציה המקבלת תמונה ומסכה ועושה בינהם קונבולוציה
    reversemask =reverse_mask(mask)   # הפיכת המסכה
    if image.dtype == np.uint8:           # התאמת סוג המסכה והתמונה
        filtered_image = centered_correlation(image.astype(np.float32), reversemask.astype(np.float32))
        filtered_image = filtered_image.astype(np.uint8)
    elif image.dtype == np.uint16:
        filtered_image = centered_correlation(image.astype(np.float32), reversemask.astype(np.float32))
        filtered_image = filtered_image.astype(np.uint16)
    else:
        filtered_image = centered_correlation(image, reversemask) # ביצוע הקורלציה
   
    return filtered_image # החזרת התמונה המסוננת

#filtered = filter_image(face, mask)
#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(face, cmap = 'gray')
plt.title('Original')
plt.subplot(1,2,2)
#plt.imshow(filtered, cmap = 'gray')
plt.title('filter')
plt.show()

#%%
# b. 
# Convert to 4 gray levels:
    
face = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/face.tif')

Image4graylevels = face & 192 # "192 = 128 + 64 = 11000000 this is save 4 gray levels of the picture"

plt.figure()
plt.subplot(1,2,1)
plt.imshow(face, cmap='gray')
plt.title("Original image")
plt.subplot(1,2,2)
plt.imshow(Image4graylevels, cmap='gray')
plt.title('Reduce Graylevels to 4 by bitband operator')
plt.show()

# gradient by Sobel mask
# Sobel masks
sobX = np.array([[1 , 0,  -1],[2 , 0 , -2], [1 ,0, -1] ])/8
sobY = sobX.T

"power"
DFacex = filter_image(Image4graylevels, sobX) # convolution of the image with sobelx.
DFacey = filter_image(Image4graylevels, sobY) # convolution of the image with sobely.
GradientFaceSqrt = np.sqrt(DFacex**2+DFacey**2)

"absolute value"
GradientFaceAbs = np.absolute(DFacex)+np.absolute(DFacey)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(Image4graylevels, cmap='gray')
plt.title("Reduce Graylevels to 4 by bitband operator")

plt.subplot(1,3,2)
plt.imshow(GradientFaceSqrt, cmap = 'gray')
plt.title('Total Gradient by sqrt')

plt.subplot(1,3,3)
plt.imshow(GradientFaceAbs,cmap = 'gray')
plt.title('Total Gradient by abs')
plt.show()

# Cutting:
#%% 

face = plt.imread('D:/שנה ג/סמסטר ב/עיבוד תמונות ממוחשב/תרגילים/עבודת אמצע סמסטר/תמונות עבודת אמצע/face.tif')

Image4graylevels = face & 192 # "192 = 128 + 64 = 11000000 this is save 4 gray levels of the picture"

(thresh, BinaryFace) = cv2.threshold(Image4graylevels, 127, 255, cv2.THRESH_BINARY) # 127+ is converted to 255 value, and less than it convert to 0.

plt.figure()
plt.subplot(1,2,1)
plt.imshow(Image4graylevels, cmap='gray')
plt.title("Reduce Graylevels to 4 by bitband operator")

plt.subplot(1,2,2)
plt.imshow(BinaryFace, cmap = 'gray')
plt.title('BinaryFace 127+ -> 255')

