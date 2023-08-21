from numba import jit, njit, prange
import numpy as np
import functools
import time

from misc.misc import timer

'''
Library of jist accelerated matrix operations
'''
  
@njit(parallel=True)
def pseudo_inverse(w):
    '''Pseudo inverse of eigenvalues'''
    vals = np.zeros_like(w)
    for i in prange(w.shape[0]):
        for j in prange(w[i].shape[0]):
            if w[i,j] == 0:
                vals[i,j] = 0
            else:
                vals[i,j] = 1 / w[i,j]
    return vals


@njit(parallel=True)
def diagonalise(X):
    """Diagonalise an array of vectors"""    
    l, n = X.shape
    out = np.zeros((l, n, n))
    for i in prange(l):
        for j in prange(n):
            out[i, j, j] = X[i, j]
    return out

  
@njit(parallel=True)
def make_hermitian_XTX_numba(X):
    """Return an array of l times X^T X from an array of l times X 2D matrices"""
    l, m, n = X.shape
    XTX = np.zeros((l, n, n), dtype=X.dtype)
    for i in prange(l):
        for j in prange(n):
            for k in prange(n):
                sum_result = 0.0
                for r in prange(m):
                    sum_result += X[i, r, j] * X[i, r, k]
                XTX[i, j, k] = sum_result
    return XTX

@njit(parallel=True)
def make_hermitian_XXT_numba(X):
    """Return an array of l times X X^T from an array of l times X 2D matrices"""
    l, m, n = X.shape
    XXT = np.zeros((l, m, m), dtype=X.dtype)
    for i in prange(l):
        for j in prange(m):
            for k in prange(m):
                sum_result = 0.0
                for r in prange(n):
                    sum_result += X[i, j, r] * X[i, k, r]
                XXT[i, j, k] = sum_result
    return XXT

@njit(parallel=True)
def make_hermitian_XTX(X):
    """Return an array of l times X^T X from an array of l times X 2D matrices"""
    l, m, n = X.shape
    XTX = np.zeros((l, n, n), dtype=X.dtype)
    for i in prange(l):
        XTX[i] = np.dot(X[i].T, X[i])
    return XTX

@njit(parallel=True)
def make_hermitian_XXT(X):
    """Return an array of l times X X^T from an array of l times X 2D matrices"""
    l, m, n = X.shape
    XXT = np.zeros((l, m, m), dtype=X.dtype)
    for i in prange(l):
        XXT[i] = np.dot(X[i], X[i].T)
    return XXT


@njit(parallel=True)
def regularise_matrices(X, eps=1e-2):
    """Add a small value to the diagonal of an array of matrices"""
    l, n, _ = X.shape
    Y = np.zeros_like(X)
    for i in prange(l):
        for j in prange(n):
            for k in prange(n):
                if j == k:
                    Y[i, j, k] = X[i, j, k] + eps
                else:
                    Y[i, j, k] = X[i, j, k]
    return Y


from math import sqrt

  
@njit(parallel=True)
def sqrt_eigenvals(w):
    '''Square root of eigenvalues'''
    l, n = w.shape
    vals = np.zeros((l,n))
    for i in prange(l):
        for j in prange(n):
            if w[i,j] < 1e-10:
                continue
            vals[i,j] = sqrt(w[i,j])

    return vals

@njit(parallel=True)
def test_hermitian(A):
    l, n, _ = A.shape
    for i in prange(l):
        # Check if matrix is square
        if A[i].shape[0] != A[i].shape[1]:
            raise ValueError("Matrix is not square")
        # Check if matrix is Hermitian
        for j in range(n):
            for k in range(n):
                if A[i, j, k] != np.conj(A[i, k, j]):
                    raise ValueError("Matrix is not Hermitian")
    print("Matrices are Hermitian")

@njit(parallel=True)
def test_real(A):
    """ Check if matrix is real """
    l, n, _ = A.shape
    for i in prange(l):
        for j in range(n):
            for k in range(n):
                if A[i,j,k].imag != 0:
                    raise ValueError("Matrix is not real")
    print("Matrices are real")

@njit(parallel=True)
def test_spd(A):
    """ Check if matrix is symmetric positive definite """
    l, n, _ = A.shape
    for i in prange(l):
        # Check if matrix is symmetric
        for j in range(n):
            for k in range(j+1, n):
                if A[i, j, k] != A[i, k, j]:
                    raise ValueError("Matrix is not symmetric")
        # Check if matrix is positive definite
        eigvals = np.linalg.eigvals(A[i])
        for eigval in eigvals:
            if eigval <= 0:
                raise ValueError("Matrix is not positive definite")
    print("Matrices are symmetric positive definite")

@njit
def test_nan_inf(A):
    """ Check if matrix contains NaN or Inf """
    l, n, _ = A.shape
    for i in range(l):
        for j in range(n):
            for k in range(n):
                if np.isnan(A[i,j,k]) or np.isinf(A[i,j,k]):
                    raise ValueError("Matrix contains NaN or Inf")
    print("Matrices do not contain NaN or Inf")

def test_matrix(Z):
    """ Test if matrix is real, symmetric positive definite and does not contain NaN or Inf """
    test_real(Z)
    test_spd(Z)
    test_hermitian(Z)
    test_nan_inf(Z)