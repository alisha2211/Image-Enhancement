# Image-Enhancement

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('classroom.jpg')
imgg =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

HistEq = cv2.equalizeHist(gray)
binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # binarize the image
kernel = np.ones((3, 3), np.uint8) # define the kernel
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1) # opening the image

plt.subplot(2,2,1)
plt.imshow(imgg)
plt.subplot(2,2,2)
plt.imshow(gray)
plt.subplot(2,2,3)
plt.imshow(HistEq)
plt.subplot(2,2,4)
plt.imshow(opening)
