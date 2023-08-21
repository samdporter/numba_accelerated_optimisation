import numpy as np
from numba import njit, jit, prange
from KoppEigen import *
from VTV import *
from MatrixOperations import *

def arrayOfMatrices(n, m, x):
    '''Generate an array of x random matrices of size n x m'''
    return np.array([np.random.rand(n, m) for _ in range(x)])

def arrayOfMatrices_with_negative_values(n, m, x):
    '''Generate an array of x random matrices of size n x m'''
    return np.array([np.random.rand(n, m) - 0.5 for _ in range(x)])

# Test numpy and Kopp implementation of eigenvalues and eigenvectors

n = 3
m = 4
x = 1000
A = arrayOfMatrices(n,m,x)
Z = make_hermitian_XTX(A)

print(Z.shape)

def test_hermitian():
    for i in range(len(Z)):
        assert np.allclose(Z[i], Z[i].T.conj())
    assert(Z.shape == (x, n, n))

val_Kopp, vec_Kopp = eigen_array(Z, eigen_kopp)
val_numpy, vec_numpy = eigen_array(Z, eigen_numpy)
val_numba, vec_numba = numba_eigen_array(Z)

def test_eigen_kopp():
    for i in range(len(Z)):
        assert np.allclose(Z[i] @ vec_Kopp[i], val_Kopp[i] * vec_Kopp[i])
        
def test_eigen_numpy():
    for i in range(len(Z)):
        assert np.allclose(Z[i] @ vec_numpy[i], val_numpy[i] * vec_numpy[i])
        
def test_eigen_numba():
    for i in range(len(Z)):
        assert np.allclose(Z[i] @ vec_numba[i], val_numba[i] * vec_numba[i])

def test_all_positive():
    for i in range(len(Z)):
        assert np.all(val_Kopp[i] > 0)
        assert np.all(val_numpy[i] > 0)
        assert np.all(val_numba[i] > 0)
