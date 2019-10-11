import numpy as np
import cv2
from matplotlib import pyplot as plt


imagen = cv2.imread('DSC03932.jpg', 1)

rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

lower_n = np.array([100,130,90])
upper_n = np.array([150,200,160])

mask = cv2.inRange(lab,lower_n,upper_n)
k = np.ones((5,5),np.uint8)
mask = cv2.erode(mask,k)
mask = cv2.dilate(mask,k,iterations = 10)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)



segm  = cv2.bitwise_and(rgb, rgb, mask = mask)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.imshow(rgb)
ax0.set_title('RGB')
ax1.imshow(lab)
ax1.set_title('LAB')
ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')
ax3.imshow(segm)
ax3.set_title('BGR')
plt.show()
