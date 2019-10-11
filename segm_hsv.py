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


lower_n = np.array([130,130,130])
upper_n = np.array([150,200,160])

n = cv2.inRange(hsv,lower_n,upper_n)

segm  = cv2.bitwise_and(rgb, rgb, mask = n)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.imshow(rgb)
ax0.set_title('RGB')
ax1.imshow(hsv)
ax1.set_title('HSV')
ax2.imshow(n, cmap='gray')
ax2.set_title('Mask')
ax3.imshow(segm)
ax3.set_title('BGR')
plt.show()
