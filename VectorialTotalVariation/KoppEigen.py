import math
import numpy as np

from numba import jit, njit, prange

import functools
import time

from misc.misc import timer

from VectorialTotalVariation.Kopp_C._kopp import zheevh3


@njit(parallel=True)
def eigenvals_cordana(Z):
    ''' jit accelerated eigenvalue calculation using Cordona method'''
    out_vals = np.zeros(Z.shape, dtype=np.complex128)
    for i in prange(Z.shape[0]):
        v = Z[i]
        a, b, c, d, e, f = v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2]

        # Analytic eigenvalues solution of the 3x3 input matrix
        tmp1 = -a**2 + a*b + a*c - b**2 + b*c - c**2 - 3*d**2 - 3*e**2 - 3*f**2
        tmp2 = 2*a**3 - 3*a**2*b - 3*a**2*c - 3*a*b**2 + 12*a*b*c - 3*a*c**2 + 9*a*d**2 + 9*a*e**2 - 18*a*f**2 + 2*b**3 - \
            3*b**2*c - 3*b*c**2 + 9*b*d**2 - 18*b*e**2 + 9*b*f**2 + \
            2*c**3 - 18*c*d**2 + 9*c*e**2 + 9*c*f**2 + 54*d*e*f
        tmp3 = np.sqrt((4*tmp1**3 + tmp2**2) + 0j)
        tmp4 = (tmp2 + tmp3) ** (1/3)
        tmp5 = 1/3*(a + b + c)
        tmp6 = 1 + 1j*np.sqrt(3)
        tmp7 = 1 - 1j*np.sqrt(3)
        eigv1 = tmp4/(3*2**(1/3)) - (2**(1/3)*tmp1)/(3*tmp4) + tmp5
        eigv2 = (tmp6*tmp1)/(3*2**(2/3)*tmp4) - (tmp7*tmp4)/(6*2**(1/3)) + tmp5
        eigv3 = (tmp7*tmp1)/(3*2**(2/3)*tmp4) - (tmp6*tmp4)/(6*2**(1/3)) + tmp5

        # Assume the values are real ones and remove the FP rounding errors
        eigv1 = np.real(eigv1)
        eigv2 = np.real(eigv2)
        eigv3 = np.real(eigv3)

        # Sort the eigenvalues using a fast sorting network
        eigv1, eigv2 = min(eigv1, eigv2), max(eigv1, eigv2)
        eigv2, eigv3 = min(eigv2, eigv3), max(eigv2, eigv3)
        eigv1, eigv2 = min(eigv1, eigv2), max(eigv1, eigv2)

        out_vals[i] = np.array([eigv1, eigv2, eigv3])

    return out_vals

from math import sqrt

@njit(parallel=True, fastmath=True)
def eigen_cordana(Z):
    ''' jit accelerated eigen vector calculation using Cordona method'''
    out_vecs = np.zeros(Z.shape, dtype=np.complex128)
    out_vals = np.zeros(Z.shape[:1] + (3,), dtype=np.float64)
    
    for i in prange(Z.shape[0]):
        v = Z[i]
        a, b, c, d, e, f = v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2]

        # Analytic eigenvalues solution of the 3x3 input matrix
        tmp1 = -a**2 + a*b + a*c - b**2 + b*c - c**2 - 3*d**2 - 3*e**2 - 3*f**2
        tmp2 = 2*a**3 - 3*a**2*b - 3*a**2*c - 3*a*b**2 + 12*a*b*c - 3*a*c**2 + 9*a*d**2 + 9*a*e**2 - 18*a*f**2 + 2*b**3 - \
            3*b**2*c - 3*b*c**2 + 9*b*d**2 - 18*b*e**2 + 9*b*f**2 + \
            2*c**3 - 18*c*d**2 + 9*c*e**2 + 9*c*f**2 + 54*d*e*f
        tmp3 = np.sqrt((4*tmp1**3 + tmp2**2))
        tmp4 = (tmp2 + tmp3) ** (1/3)
        tmp5 = 1/3*(a + b + c)
        tmp6 = 1 + 1j*np.sqrt(3)
        tmp7 = 1 - 1j*np.sqrt(3)
        eigv1 = tmp4/(3*2**(1/3)) - (2**(1/3)*tmp1)/(3*tmp4) + tmp5
        eigv2 = (tmp6*tmp1)/(3*2**(2/3)*tmp4) - (tmp7*tmp4)/(6*2**(1/3)) + tmp5
        eigv3 = (tmp7*tmp1)/(3*2**(2/3)*tmp4) - (tmp6*tmp4)/(6*2**(1/3)) + tmp5

        # Assume the values are real ones and remove the FP rounding errors
        eigv1 = np.real(eigv1)
        eigv2 = np.real(eigv2)
        eigv3 = np.real(eigv3)

        # Sort the eigenvalues using a fast sorting network
        eigv1, eigv2 = min(eigv1, eigv2), max(eigv1, eigv2)
        eigv2, eigv3 = min(eigv2, eigv3), max(eigv2, eigv3)
        eigv1, eigv2 = min(eigv1, eigv2), max(eigv1, eigv2)

        w = np.array([eigv1, eigv2, eigv3])

        Q = np.zeros((3, 3), dtype=np.complex128)
        Q[0, 1] = v[0, 1]*v[1, 2] - v[0, 2]*v[1, 1]
        Q[1, 1] = v[0, 2]*np.conj(v[0, 1]) - v[1, 2]*v[0, 0]
        Q[2, 1] = np.abs(v[0, 1])**2 

        # Calculate first eigenvector
        Q[0, 0] = Q[0, 1] + v[0, 2]*w[0]
        Q[1, 0] = Q[1, 1] + v[1, 2]*w[0]
        Q[2, 0] = (v[0, 0] - w[0]) * (v[1, 1] - w[0]) - Q[2, 1]

        # Calculate second eigenvector
        Q[0, 1] = Q[0, 1] + v[0, 2]*w[1]
        Q[1, 1] = Q[1, 1] + v[1, 2]*w[1]
        Q[2, 1] = (v[0, 0] - w[1]) * (v[1, 1] - w[1]) - Q[2, 1]

        # Calculate third eigenvector
        Q[0, 2] = np.conj(Q[1, 0]*Q[2, 1] - Q[2, 0]*Q[1, 1])
        Q[1, 2] = np.conj(Q[2, 0]*Q[0, 1] - Q[0, 0]*Q[2, 1])
        Q[2, 2] = np.conj(Q[0, 0]*Q[1, 1] - Q[1, 0]*Q[0, 1])

        out_vecs[i] = Q
        Q = np.real(Q)
        out_vals[i] = w

    return out_vals, out_vecs
 
@jit(nopython=True, parallel=True)
def eigen_2x2(arr):
    n = arr.shape[0]
    eigenvalues = np.empty((n, 2), dtype=arr.dtype)
    eigenvectors = np.empty((n, 2, 2), dtype=arr.dtype)
    for i in prange(n):
        a, b, c, d = arr[i].ravel()
        sqrt_term = np.sqrt((a - d)**2 + 4*b*c)
        lambda1 = (a + d + sqrt_term) / 2
        lambda2 = (a + d - sqrt_term) / 2
        eigenvalues[i, 0] = lambda1
        eigenvalues[i, 1] = lambda2

        # First eigenvector
        if b != 0:
            eigenvectors[i, 0, 0] = -b / (a - lambda1)
            eigenvectors[i, 0, 1] = 1
        else:
            eigenvectors[i, 0, 0] = 1
            eigenvectors[i, 0, 1] = 0
        
        # Second eigenvector
        if b != 0:
            eigenvectors[i, 1, 0] = -b / (a - lambda2)
            eigenvectors[i, 1, 1] = 1
        else:
            eigenvectors[i, 1, 0] = 1
            eigenvectors[i, 1, 1] = 0

        # Normalize each eigenvector individually
        eigenvectors[i, 0] = eigenvectors[i, 0] / np.linalg.norm(eigenvectors[i, 0])
        eigenvectors[i, 1] = eigenvectors[i, 1] / np.linalg.norm(eigenvectors[i, 1])

    return eigenvalues, eigenvectors

@njit(parallel=True)
def eigenvals_2x2(arr):
    n = arr.shape[0]
    eigenvalues = np.empty((n, 2), dtype=arr.dtype)
    for i in prange(n):
        a, b, c, d = arr[i].ravel()
        sqrt_term = np.sqrt((a - d)**2 + 4*b*c)
        lambda1 = (a + d + sqrt_term) / 2
        lambda2 = (a + d - sqrt_term) / 2
        eigenvalues[i, 0] = lambda1
        eigenvalues[i, 1] = lambda2
    return eigenvalues
    

@jit(forceobj=True)
def eigen_numpy(x):
    """ Uses numpy's linalg.eig
    returns two numpy arrays in np.linalg.eig order
    [0] = Eigenvalues
    [1] = Eigenvectors
    """
    return np.linalg.eig(x)


@jit(forceobj=True)
def eigen_kopp(x):
    """ Uses Kopp's C implementation wrapped with SWIG
    returns two numpy arrays in np.linalg.eig order
    [0] = Eigenvalues
    [1] = Eigenvectors
    """
    return zheevh3(x)[1:][::-1]

# jitify the function


@jit(forceobj=True)
def eigen_array(A, func):
    """ Calculate the eigenvalues and eigenvectors of a 3x3 matrix using a given function """
    val = np.zeros(A.shape[0:2])
    vec = np.zeros(A.shape)

    for i in prange(A.shape[0]):
        val[i], vec[i] = func(A[i])
        val[i], vec[i] = np.real(val[i]), np.real(vec[i])

    return val, vec


@jit(forceobj=True)
def eigenvals_cordana_array(Z, out):
    """ Calculate the eigenvalues of a 3x3 matrix using Cordana's method """
    for i in prange(Z.shape[0]):
        out[i] = eigenvals_cordana(Z[i])
    return out


@njit(parallel=True)
def truncate_small_negatives(A, threshold=1e-3):
    """ Truncate small negative values to zero """
    out = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            if A[i, j] < -threshold:
                print("found eigen value of ", (A[i, j]))
                Warning("Negative value below threshold. Setting to zero, but something is wrong.")
                out[i, j] = 0
            if A[i, j] < 0 and A[i, j] > -threshold:
                out[i, j] = 0
            else:
                out[i, j] = A[i, j]
    return out


@njit(parallel=True)
def numba_eigen_array(A):
    """ Calculate the eigenvalues and eigenvectors of a 3x3 matrix using a given function """
    val = np.zeros(A.shape[0:2])
    vec = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        val[i], vec[i] = np.linalg.eig(A[i])
        # sort eigenvalues and eigenvectors in ascending eigenvalue order
        idx = val[i].argsort()[::-1]
        val[i] = val[i][idx]
        vec[i] = vec[i][:, idx]
    return val, vec


@njit(parallel=True)
def numba_eigenvals_array(A):
    """ Calculate the eigenvalues of a 3x3 matrix using a given function """
    val = np.zeros(A.shape[0:2])
    for i in prange(A.shape[0]):
        val[i] = np.linalg.eigvals(A[i])
        # sort eigenvalues in ascending order
        idx = val[i].argsort()[::-1]
        val[i] = val[i][idx]
    return val

@njit
def normalise_eigen_array(A):
    l, n, _ = A.shape
    eigvecs_norm = np.empty_like(A)
    vals = np.empty((l, n))
    for i in range(l):
        val, vecs = np.linalg.eig(A[i])
        # Ensure eigenvectors are normalized
        for j in range(n):
            vecs[:, j] = vecs[:, j] / np.linalg.norm(vecs[:, j])
        eigvecs_norm[i] = vecs
        vals[i] = val
    return vals, eigvecs_norm
