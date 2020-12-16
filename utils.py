import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.signal import convolve
import math

def readrgb_double(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0

def rgb2gray(rgb):
    """
    rgb: [0-1]
    """
    return rgb[...,0]*0.299 + 0.587*rgb[...,1] + 0.114*rgb[...,2]

def showgray(img, vmax=1):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def show(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
def plotmxn(m, n, data, is_image, labels=None,  size=(10, 10)):
    if size:
        plt.figure(figsize=size)
    for i in range(len(data)):
        ax = plt.subplot(m,n,i+1)
    #     plt.grid(False)
        if is_image:
            plt.imshow(data[i])
        else:
            plt.plot(data[i][0], data[i][1])
        if labels:
            plt.xlabel(labels[i])
    plt.show()