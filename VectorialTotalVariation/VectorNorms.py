import numpy as np
from numba import njit, prange

from misc.misc import timer

 
@njit(parallel=True)
def prox_l1(x, lam):
    """
    Proximal operator for the L1 norm
    """
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            sign = np.sign(x[i, j])
            tmp = np.abs(x[i, j]) - lam
            max = np.maximum(0.0, tmp)
            res[i, j] = sign * max
    return res
 
@njit(parallel=True)
def prox_l2(x, lam, eps= 1e-8):
    """
    Proximal operator for the L2 norm
    """
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        norm_xi = np.linalg.norm(x[i, :])
        tmp =  1.0 - lam / (norm_xi+eps)
        max = np.maximum(0.0, tmp)
        res[i, :] = max * x[i, :]
    return res

@njit(parallel=True)
def prox_linf_old(x, lam):
    """
    Proximal operator for the L-infinity norm
    """
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        tmp = x[i]/lam
        norm = np.abs(tmp).sum()
        if norm > 1:
            tmp/=tmp.sum()
        res[i] = x[i] - lam * tmp
    return res

@njit(parallel=True)
def prox_linf(x, lam):
    """
    Proximal operator for the L-infinity norm
    """
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        res[i, :] = np.sign(x[i, :]) * np.minimum(np.abs(x[i, :]), lam)
    return res


@njit(parallel=True)
def l1_norm(arr):
    n, l = arr.shape
    result = np.zeros(n)
    for i in prange(n):
        for j in range(l):
            result[i] += abs(arr[i][j])
    return result

from math import sqrt

@njit(parallel=True)
def l2_norm(arr):
    n, l = arr.shape
    result = np.zeros(n)
    for i in prange(n):
        for j in range(l):
            result[i] += arr[i][j]**2
        result[i] = sqrt(result[i])
    return result

@njit(parallel=True)
def linf_norm(arr):
    n, l = arr.shape
    result = np.zeros(n)
    for i in prange(n):
        max_val = abs(arr[i][0])
        for j in range(1, l):
            max_val = max(max_val, abs(arr[i][j]))
        result[i] = max_val
    return result
