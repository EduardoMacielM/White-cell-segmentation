import numpy as np
import cv2
import easygui
from matplotlib import pyplot as plt


path = easygui.fileopenbox()
imagen = cv2.imread(path,1)
height = np.shape(imagen)[0]
width = np.shape(imagen)[1]
rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

red = rgb[:,:,0]
green = rgb[:,:,1]
blue = rgb[:,:,2]
r = np.ones((height,width))
g = np.ones((height,width))
b = np.ones((height,width))
for x in range (0,height):
    for y in range (0,width):
        if red[x,y] > 140 & red[x,y] < 160:
            r[x,y] = 0
        if green[x,y] > 140 & green[x,y] < 160:
            g[x,y] = 0
        if blue[x,y] > 140 & blue[x,y] < 160:
            b[x,y] = 0
mask= r*g*b
        
fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.imshow(rgb)
ax0.set_title('RGB')
ax1.imshow(hsv)
ax1.set_title('HSV')
ax2.imshow(gray, cmap='gray')
ax2.set_title('Grayscale')
ax3.imshow(bgr)
ax3.set_title('segmented RGB')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.imshow(r,'gray')
ax0.set_title('RGB')
ax1.imshow(g,'gray')
ax1.set_title('HSV')
ax2.imshow(g,'gray')
ax2.set_title('Grayscale')
ax3.imshow(mask,'gray')
ax3.set_title('segmented RGB')
plt.show()
