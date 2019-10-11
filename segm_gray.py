import numpy as np
import cv2
import easygui
from matplotlib import pyplot as plt


path = '/home/eduardom/Pictures/celulas/DSC03947.JPG'
imagen = cv2.imread(path,1)
height = np.shape(imagen)[0]
width = np.shape(imagen)[1]
rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

ret,mask = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
k = np.ones((5,5),np.uint8)
mask = cv2.erode(mask,k,iterations = 7)
mask = cv2.dilate(mask,k,iterations = 10)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)

segm  = cv2.bitwise_and(rgb, rgb, mask = mask)


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.imshow(rgb)
ax0.set_title('RGB')
ax1.imshow(mask,'gray')
ax1.set_title('HSV')
ax2.imshow(gray, cmap='gray')
ax2.set_title('Grayscale')
ax3.imshow(segm)
ax3.set_title('BGR')
plt.show()
