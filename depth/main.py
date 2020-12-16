#%%
%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
from depth.akai import *
from utils import *
from skimage import feature
from depth import pipeline
from defocus_estimation import zhuo
from skimage import color
import scipy.ndimage as sciimg
import scipy.signal as scisig
#%%
I = np.arange(0,27).reshape((3,3,3))
I = np.transpose(I, (2, 1, 0)) / 27
s = np.arange(0,9).reshape((3,3)).T/9
sd=1
sr=0.1
#%%
ib, id =  zhuo.sparse_bilateral_filter(s, I, sd, sr)

# %%
