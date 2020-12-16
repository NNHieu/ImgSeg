
#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt 
#%%
img = cv2.imread("./test2.jpg", 0)
equ = cv2.equalizeHist(img)
plt.hist(img.flat, bins=100, range=(0, 255))
# %%
