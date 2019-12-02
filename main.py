import blobsNeut

import blobsLinf
import blobsMon

import numpy as np
import cv2
from matplotlib import pyplot as plt

path = '18.jpg'

blobsNeut.neutro(path)
blobsLinf.linf(path)
blobsMon.mon(path)
