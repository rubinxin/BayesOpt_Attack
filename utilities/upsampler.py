# @author: Robin Ru (robin@robots.ox.ac.uk)

import numpy as np
import torch
import torch.nn.functional as F


def upsample_projection(dim_reduction, X_low, low_dim, high_dim, nchannel=1, align_corners=True):
    """
    Various upsampling methods: CLUSTER,  BILI (bilinear), NN (bilinear), BICU (bicubic)

    :param dim_reduction: dimension reduction method used in upsampling
    :param X_low: input data in low dimension
    :param low_dim: the low dimension
    :param high_dim: the high dimension
    :param nchannel: number of image channels
    :param align_corners: align corner option for interpolate
    :return X_high: input data in high dimension
    """

    if dim_reduction == 'CLUSTER':
        n = X_low.shape[0]
        high_int = int(np.sqrt(high_dim))
        low_int = int(np.sqrt(low_dim))
        ratio = np.floor(high_dim / low_dim)
        low_edge = int(np.floor(np.sqrt(ratio)) + 1)
        high_edge = low_edge * low_int
        high_obs = np.zeros((n, high_edge, high_edge, nchannel))
        for ch in range(nchannel):
            for row in range(low_int):
                for col in range(low_int):
                    k = row * low_int + col
                    for jj in range(low_edge):
                        for ii in range(low_edge):
                            high_obs[:, row * low_edge + jj, col * low_edge + ii, ch] = X_low[:, ch * low_dim + k]

        high_obs = high_obs[:, :high_int, :high_int, :]
        X_high = high_obs.reshape(X_low.shape[0], high_dim * nchannel)

    else:

        if dim_reduction == 'BILI':
            upsample_mode = 'bilinear'
        elif dim_reduction == 'NN':
            upsample_mode = 'nearest'
            align_corners = None
        elif dim_reduction == 'BICU':
            upsample_mode = 'bicubic'

        X_low_tensor = torch.FloatTensor(X_low).view(X_low.shape[0], nchannel, int(np.sqrt(low_dim)),
                                                     int(np.sqrt(low_dim)))
        X_high_tensor_resize = F.interpolate(X_low_tensor,
                                             size=(int(np.sqrt(high_dim)), int(np.sqrt(high_dim))), mode=upsample_mode,
                                             align_corners=align_corners)
        X_high = X_high_tensor_resize.data.numpy().squeeze().reshape(X_low.shape[0], high_dim * nchannel)
    return X_high


def downsample_projection(dim_reduction, X_high, low_dim, high_dim, nchannel=1, align_corners=True):
    """
    Various downsampling methods: CLUSTER,  BILI (bilinear), NN (bilinear), BICU (bicubic)

    :param dim_reduction: dimension reduction method used in upsampling
    :param X_high: input data in high dimension
    :param low_dim: the low dimension
    :param high_dim: the high dimension
    :param nchannel: number of image channels
    :param align_corners: align corner option for interpolate
    :return X_low: input data in low dimension
    """

    if dim_reduction == 'CLUSTER':

        n = X_high.shape[0]
        high_int = int(np.sqrt(high_dim))
        X_high = np.reshape(X_high, (n, high_int, high_int, nchannel))
        low_int = int(np.sqrt(low_dim))
        ratio = np.floor(high_dim / low_dim)
        low_edge = int(np.floor(np.sqrt(ratio)) + 1)
        high_edge = low_edge * low_int
        npadd = high_edge - high_int
        high_obs = np.zeros((n, high_edge, high_edge, nchannel))
        high_obs[:, :high_int, :high_int, :] = X_high
        for kk in range(npadd):
            high_obs[:, high_int + kk, :high_int, :] = high_obs[:, high_int - 1, :high_int, :]
        for kk in range(npadd):
            high_obs[:, :, high_int + kk, :] = high_obs[:, :, high_int - 1, :]
        low_obs = np.zeros((n, low_int, low_int, nchannel))
        for ch in range(nchannel):
            for row in range(low_int):
                for col in range(low_int):
                    for point in range(n):
                        low_obs[point, row, col, ch] = np.mean(
                            high_obs[point, (row * low_edge):((row + 1) * low_edge),
                            (col * low_edge):((col + 1) * low_edge), ch])
        X_low_tensor_resize = low_obs
        X_low = X_low_tensor_resize.reshape(X_high.shape[0], low_dim * nchannel)

    else:
        if dim_reduction == 'BILI':
            upsample_mode = 'bilinear'
        elif dim_reduction == 'NN':
            upsample_mode = 'nearest'
            align_corners = None
        elif dim_reduction == 'BICU':
            upsample_mode = 'bicubic'

        X_high_tensor = torch.FloatTensor(X_high).view(X_high.shape[0], nchannel, int(np.sqrt(high_dim)),
                                                       int(np.sqrt(high_dim)))
        X_low_tensor_resize = F.interpolate(X_high_tensor,
                                            size=(int(np.sqrt(low_dim)), int(np.sqrt(low_dim))), mode=upsample_mode,
                                            align_corners=align_corners)
        X_low = X_low_tensor_resize.data.numpy().squeeze().reshape(X_high.shape[0], low_dim * nchannel)
    return X_low
