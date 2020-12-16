import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
from skimage import feature
from matting import *


def g1x(x, y, s1):
    '''
    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(x, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g


def g1y(x, y, s1):
    '''
    :param x:
    :param y:
    :param s1:
    :return:
    '''
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(y, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g


def estimate_sparse_blur(gimg, edge_map, std1, std2):
    '''
    :param gimg: Grayscale image
    :param edge_map: An edge map of the image
    :param std1: Standard deviation of reblurring
    :param std2: Standard deviation of second reblurring
    :return: Estimated sparse blur values at edge locations
    '''
    half_window = 11
    m = half_window * 2 + 1
    a = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(a, (m, 1))
    ymesh = xmesh.T

    f11 = g1x(xmesh, ymesh, std1)
    f12 = g1y(xmesh, ymesh, std1)

    f21 = g1x(xmesh, ymesh, std2)
    f22 = g1y(xmesh, ymesh, std2)

    gimx1 = scipy.ndimage.convolve(gimg, f11, mode='nearest')
    gimy1 = scipy.ndimage.convolve(gimg, f12, mode='nearest')
    mg1 = np.sqrt(gimx1 ** 2 + gimy1 ** 2)

    gimx2 = scipy.ndimage.convolve(gimg, f21, mode='nearest')
    gimy2 = scipy.ndimage.convolve(gimg, f22, mode='nearest')
    mg2 = np.sqrt(gimx2 ** 2 + gimy2 ** 2)

    R = np.divide(mg1, mg2)
    R = np.multiply(R, edge_map > 0)

    sparse_vals = np.divide(R ** 2 * (std1 ** 2) - (std2 ** 2), 1 - R ** 2)
    sparse_vals[sparse_vals < 0] = 0

    sparse_bmap = np.sqrt(sparse_vals)
    sparse_bmap[np.isnan(sparse_bmap)] = 0
    sparse_bmap[sparse_bmap > 5] = 5

    return sparse_bmap


def estimate_bmap_laplacian(img, sigma_c, std1, std2):
    '''
    :param img: An RGB image [0-255]
    :param sigma_c: Sigma parameter for Canny edge detector
    :param std1: Standard deviation of reblurring
    :param std2: Standard deviation of second reblurring
    :return: defocus blur map of the given image
    '''
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    edge_map = feature.canny(gimg, sigma_c)

    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1, std2)
    h, w = sparse_bmap.shape

    L1 = get_laplacian(img / 255.0)
    A, b = make_system(L1, sparse_bmap.T)

    bmap = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T

    return bmap