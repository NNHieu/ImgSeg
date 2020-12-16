from math import sqrt
from utils import show, showgray
import cv2
import numpy as np
import scipy.ndimage as sciimg
from filters.gaussian import matlab_style_gauss2D
from filters.bilateral_filters import sparse_bilateral_filter
from utils import *

def g1x(x, y, sigma):
    ssq = sigma ** 2
    return -x/(2*np.pi*ssq**2)*np.exp(-(x**2 + y**2)/(2*ssq)) 

def g1y(x, y, sigma):
    ssq = sigma ** 2
    return -y/(2*np.pi*ssq**2)*np.exp(-(x**2 + y**2)/(2*ssq)) 

def grad_map(gray, std):
    half_window = (2*np.ceil(2*std)) + 1
    a = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(a, (a.shape[0], 1))
    ymesh = xmesh.T

    gx = g1x(xmesh, ymesh, std)
    gy = g1y(xmesh, ymesh, std)
    gimx = sciimg.convolve(gray, gx, mode='nearest')
    gimy = sciimg.convolve(gray, gy, mode='nearest')
    return np.sqrt(gimx**2 + gimy**2)

def sparse_blur_map(gray, edge_map, std, max_blur):
    std1 = std
    std2 = 1.5*std
    mg1 = grad_map(gray, std1)
    mg2 = grad_map(gray, std2)
    g_ratio = mg1/mg2

    sparse_map = (g_ratio**2*std1**2-std2**2)/(1-g_ratio**2)
    sparse_map[sparse_map < 0] = 0
    sparse_map[edge_map == 0] = 0
    sparse_map = np.sqrt(sparse_map)
    sparse_map[sparse_map > max_blur] = max_blur
    return sparse_map

def defocus_estimation(image, edge_map, std, lambd, max_blur):
    """
    image: [0-1]
    """
    gray = rgb2gray(image)
    sparse_map = sparse_blur_map(gray, edge_map, std, max_blur)
    sd = 5   #spatial sigma
    sr = 0.1 #range sigma
    ssparse_map,_ =sparse_bilateral_filter(sparse_map, image, sd, sr);
    return sparse_map, ssparse_map

