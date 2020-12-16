import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
from skimage import feature
from matting import *

def gradx_gfilter(x, y, s1):
    '''
    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    # g = -1 * np.multiply(np.divide(x,np.sqrt(2 * np.pi) * s1sq ** 3),
    #                      np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g

def grady_gfilter(x, y, s1):
    '''
    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    # g = -1 * np.multiply(np.divide(y,np.sqrt(2 * np.pi) * s1sq ** 3),
    #                      np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    g = -1 * np.multiply(np.divide(y,s1sq ** 3),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h