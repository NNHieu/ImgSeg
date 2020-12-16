import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg


from skimage import feature


def make_system(L, sparse_map, constraint_factor=0.001):
    '''
    :param L: Laplacian Matrix.
    Check the paper "A closed-form solution to natural image matting"
    :param sparse_map: Estimated sparse blur values
    :param constraint_factor: Internal parameter for propagation
    :return: System parameters to solve for defocus blur propagation
    '''
    spflatten = sparse_map.ravel()

    D = scipy.sparse.diags(spflatten)

    # combine constraints and graph laplacian
    A = constraint_factor * D + L
    # constrained values of known alpha values
    b = constraint_factor * D * spflatten

    return A, b

def get_laplacian(I, r=1, eps=1e-7):
    '''
    :param I: An RGB image [0-1]
    :param r: radius
    :param eps: epsilon
    :return: The Laplacian matrix explained in Levin's paper
    '''
    h, w, c = I.shape
    wr = (2 * r + 1) * (2 * r + 1)

    M_idx = np.arange(h * w).reshape(w, h).T
    n_vals = (w - 2 * r) * (h - 2 * r) * wr ** 2

    # data for matting laplacian in coordinate form
    row_idx = np.zeros(n_vals, dtype=np.int64)
    col_idx = np.zeros(n_vals, dtype=np.int64)
    vals = np.zeros(n_vals, dtype=np.float64)
    lenr = 0

    for j in range(r, h - r):
        for i in range(r, w - r):
            winr = I[j - r:j + r + 1, i - r:i + r + 1, 2]
            wing = I[j - r:j + r + 1, i - r:i + r + 1, 1]
            winb = I[j - r:j + r + 1, i - r:i + r + 1, 0]
            win_idx = M_idx[j - r:j + r + 1, i - r:i + r + 1].T.ravel()

            meanwinr = winr.mean()
            winrsq = np.multiply(winr, winr)
            varI_rr = winrsq.sum() / wr - meanwinr ** 2

            meanwing = wing.mean()
            wingsq = np.multiply(wing, wing)
            varI_gg = wingsq.sum() / wr - meanwing ** 2

            meanwinb = winb.mean()
            winbsq = np.multiply(winb, winb)
            varI_bb = winbsq.sum() / wr - meanwinb ** 2

            winrgsq = np.multiply(winr, wing)
            varI_rg = winrgsq.sum() / wr - meanwinr * meanwing

            winrbsq = np.multiply(winr, winb)
            varI_rb = winrbsq.sum() / wr - meanwinr * meanwinb

            wingbsq = np.multiply(wing, winb)
            varI_gb = wingbsq.sum() / wr - meanwing * meanwinb

            Sigma = np.array([[varI_rr, varI_rg, varI_rb],
                              [varI_rg, varI_gg, varI_gb],
                              [varI_rb, varI_gb, varI_bb]])

            meanI = np.array([meanwinr, meanwing, meanwinb])

            Sigma = Sigma + eps * np.eye(3)

            winI = np.zeros((wr, c))

            winI[:, 0] = winr.T.ravel()
            winI[:, 1] = wing.T.ravel()
            winI[:, 2] = winb.T.ravel()

            winI = winI - meanI

            inv_cov = np.linalg.inv(Sigma)
            tvals = (1 + np.matmul(np.matmul(winI, inv_cov), winI.T)) / wr

            row_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (1, wr)).ravel()
            col_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (wr, 1)).T.ravel()
            vals[lenr:wr ** 2 + lenr] = tvals.T.ravel()

            lenr += wr ** 2

    # Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(h*w, h*w))
    Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(w * h, w * h))

    row_idx2 = np.zeros(w * h, dtype=np.int64)
    col_idx2 = np.zeros(w * h, dtype=np.int64)
    vals2 = np.zeros(w * h, dtype=np.float64)

    row_idx2[:] = np.arange(w * h)
    col_idx2[:] = np.arange(w * h)
    vals2[:] = Lsparse.sum(axis=1).ravel()

    LDsparse = scipy.sparse.coo_matrix((vals2, (row_idx2, col_idx2)), shape=(w * h, w * h))

    return LDsparse - Lsparse