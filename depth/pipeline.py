import cv2
import numpy as np
from numpy.lib.type_check import imag
from scipy import sparse
from filters.gaussian import gradx_gfilter, grady_gfilter
import scipy
from matting import *
from defocus_estimation import zhuo
from utils import *
from filters.bilateral_filters import sparse_bilateral_filter
from filters.guided_filter.gf import guided_filter

def CLAHE(img):
    '''
    param
    ------------
    img: An GRAY image [0-1]
    '''
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(img)
    return cl_img

def propagate_map(img, sparse_bmap):
    h, w = sparse_bmap.shape
    L1 = get_laplacian(img)
    A, b = make_system(L1, sparse_bmap.T)
    bmap = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T
    return bmap

def get_bmap(img, std, sigma_canny=1, max_blur=3, guide=False):
    '''
    :param img: An RGB image [0-1]
    :param sigma_c: Sigma parameter for Canny edge detector
    :param std1: Standard deviation of reblurring
    :param std2: Standard deviation of second reblurring
    :return: defocus blur map of the given image
    '''
    gray = rgb2gray(img)
    gray = CLAHE(np.uint8(gray*255))/255
    edge_map = feature.canny(gray, sigma_canny)
    sparse_map = zhuo.sparse_blur_map(gray, edge_map, std, max_blur)
    if guide:
        guided_sparse_map = guided_filter(gray, sparse_map, 5, 0.01)
    else:
        guided_sparse_map, _ = sparse_bilateral_filter(sparse_map, img, 5, 0.1)
    return propagate_map(img, guided_sparse_map)

