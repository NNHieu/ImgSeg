from math import sqrt
from utils import show, showgray
import numpy as np
import scipy.ndimage as sciimg
from filters.gaussian import matlab_style_gauss2D
from utils import *


def sparse_bilateral_filter(I, G, sd, sr, thresh=0.0001):
    if len(I.shape) == 3:
        h,w,c = I.shape
    else:
        h, w = I.shape
        c = 1
    hg, wg, cg = G.shape
    oriI = np.copy(I)
    Ib = np.zeros((h, w, c))
    wr = np.ceil(sd*4) + 1
    f = matlab_style_gauss2D((wr, wr), sd)
    pnum = int((wr - 1)/2)
    pG = np.lib.pad(G, ((pnum, pnum), (pnum, pnum), (0, 0)), mode='edge')
    for cc in range(c):
        if len(I.shape) == 3:
            pI = np.lib.pad(I[..., cc], pnum, mode='edge')
        else:
            pI = np.lib.pad(I, pnum, mode='edge')
        for jj in range(pnum, pnum + h):
            for kk in range(pnum, pnum + w):
                if I[jj - pnum, kk - pnum] > thresh:
                    # Get a local neighborhood
                    winI = pI[jj-pnum:jj+pnum + 1,kk-pnum:kk+pnum + 1]
                    winG = pG[jj-pnum:jj+pnum + 1,kk-pnum:kk+pnum + 1,:]
                    winGW = np.sum(np.exp(-(winG - winG[pnum,pnum])**2/(sr**2*2)),axis=2)
                    newW = winGW*f*np.exp(-winI)
                    index= winI > thresh
                    t = np.sum(newW[index])
                    Ib[jj-pnum,kk-pnum,cc] = np.sum(np.sum(winI[index]*newW[index]))
                    if t>0:
                        Ib[jj-pnum,kk-pnum,cc] = Ib[jj-pnum,kk-pnum,cc]/t;
    Ib = np.squeeze(Ib)
    Id = oriI / Ib
    return Ib, Id