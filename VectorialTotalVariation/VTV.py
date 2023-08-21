''' TNV functions and class'''

import numpy as np
from cil.framework import BlockDataContainer, BlockGeometry
from cil.optimisation.functions import Function
from cil.optimisation.operators import GradientOperator, LinearOperator
from numba import jit, njit, prange
from numpy import linalg as la

# make these more specific for faster import
from VectorialTotalVariation.KoppEigen import (eigen_array, eigen_kopp,
                                               eigen_2x2, eigenvals_2x2,
                                               eigen_cordana,
                                               numba_eigen_array,
                                               numba_eigenvals_array,
                                               eigenvals_cordana,
                                               truncate_small_negatives,
                                               normalise_eigen_array)
from VectorialTotalVariation.MatrixOperations import *
from VectorialTotalVariation.VectorNorms import *

import functools
import time

from numbers import Number

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time for {func}: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

@njit(parallel=True)
def _make_matrix(array, size, nchannels, ndims):
    out = np.zeros((size, nchannels, ndims,))
    for i in prange(size):
        for j in prange(nchannels):
            for k in prange(ndims):
                out[i,j,k] = array[j*ndims+k][i]
    return out

@njit(parallel=True)
def _break_matrix(array, size, nchannels, ndims,):
    out = np.zeros((nchannels * ndims, size))
    for i in prange(size):
        for j in prange(nchannels):
            for k in prange(ndims):
                out[j*ndims+k][i]= array[i,j,k] 
    return out

class BlockDataToVector():
    """
    Class to vectorise a n x m jacobian matrix.
    matrix will have structure bdc(n x bdc(m x im))
    """    
    def __init__(self, im, nchannels):
        self.image = im.clone()
        self.shape = im.shape
        self.nchannels = nchannels
        self.ndim = 2 if im.shape[0] == 1 else 3
        self.size = np.prod(im.shape)

        domain_shape = []
        self.ind = []
        for i, size in enumerate(list(im.shape) ):
            if size!=1:
                domain_shape.append(size)
                self.ind.append(i)
        # Dimension of domain geometry        
        self.ndim = len(domain_shape)

    def direct(self, blockdata):

        bdc = blockdata.clone()
        im_list = [] # can do this in init?=
        for c in bdc:
            for im in c:
                im_arr = im.as_array().flatten()
                im_list.append(im_arr)
        im_list = np.array(im_list)

        res = _make_matrix(im_list, self.size, self.nchannels, self.ndim)

        return  res
    def adjoint(self, array):
        
        out = _break_matrix(array, self.size, self.nchannels, self.ndim)
        bdc_list = []
        for i in range(out.shape[0]//self.ndim):
            bdc_list.append(BlockDataContainer(*[self.image.clone().fill(out[i*self.ndim+j].\
                                                reshape(self.image.shape)) for j in  range(self.ndim)]))

        return BlockDataContainer(*[bdc for bdc in bdc_list])

class VectorialTotalVariation(Function):
    """VNV function - atm only for 3 3D images
    Supports Nuclear or Frobenius norms

    Args:
        Function (Class): cil Function base class
    """    
    def __init__(self, domain_geometry, norm = 'nuclear',
                 static_ims = [1,2], order = 1, **kwargs):

        """ order = 0: XTX , order = 1: XXT - onlt 1 works atm"""

        super(VectorialTotalVariation, self).__init__()
        shapes = set(im.shape for im in domain_geometry.containers)
        if len(shapes) > 1:
            raise ValueError("Images not of the same shape. Not yet supported.")

        self.static_ims = static_ims

        if domain_geometry[0].shape[0] == 1:
            self.ndims = 2
        else:
            self.ndims = 3

        self.order = order
        self.n_channels = len(domain_geometry.containers)

        print(f"dims: {self.ndims}, channels: {self.n_channels}")

        if self.n_channels == 3 and self.ndims == 2:
            self.order = 0
            print("defaulting to order 0 to improved performance")
        elif self.n_channels == 2 and self.ndims == 3:
            self.order = 1
            print("defaulting to order 1 to improved performance")
        elif self.n_channels > 3 and self.ndims < 4:
            self.order = 0
            print("defaulting to order 1 otherwise impossible to compute")
        elif self.ndims > 3 and self.n_channels < 4:
            self.order = 1
            print("defaulting to order 1 otherwise impossible to compute")
        elif self.ndims == 3 and self.n_channels == 3:
            print("Either order supported")
        else:
            raise ValueError("Not yet supported for this number of channels and dimensions")
        
        self.norm = norm
            
        self.vec = BlockDataToVector(domain_geometry[0], nchannels=self.n_channels)
    
    # 
    #@jit(forceobj=True)
    def __call__(self, x):
        res = x.clone()

        jac_arr = self.vec.direct(res) # array of jacobian matrix  


        if self.order == 0:
            Z = make_hermitian_XTX(jac_arr)
        elif self.order == 1:
            Z = make_hermitian_XXT(jac_arr) # H = J J^T

        Z = regularise_matrices(Z, eps=1e-8) # Regularise matrices

        if Z.shape[1] == Z.shape[2] == 2:
            vals = eigenvals_2x2(Z)

        else:
            vals = numba_eigenvals_array(Z) # Find eigenvalues of H
        
        s = sqrt_eigenvals(vals) # Take square root of eigenvalues

        if self.norm == 'nuclear':
            res = l1_norm(s) # Sum of singular values
        elif self.norm == 'frobenius':
            res = l2_norm(s) # Frobenius norm
        return np.sum(res) # Sum over all images
    
    # 
    #@jit(forceobj=True)
    def convex_conjugate(self, x):
        res = x.clone()

        jac_arr = self.vec.direct(res) # array of jacobian matrix       

        if self.order == 0:
            Z = make_hermitian_XTX(jac_arr)
        elif self.order == 1:
            Z = make_hermitian_XXT(jac_arr) # H = J J^T

        Z = regularise_matrices(Z, eps=1e-8) # Regularise matrices

        if Z.shape[1] == Z.shape[2] == 2:
            vals= eigenvals_2x2(Z)
        else:
            vals = numba_eigenvals_array(Z) # Find eigenvalues of H
        
        s = sqrt_eigenvals(vals) # Take square root of eigenvalues

        if self.norm == 'nuclear':
            res = linf_norm(s) # Take max of eigenvalues
        elif self.norm == 'frobenius':
            res = l2_norm(s) # Take sqrt of sum of squares of eigenvalues
        return np.max(np.abs(res)) # Take max of absolute values of eigenvalues

    # 
    #@jit(forceobj=True)
    def proximal_conjugate(self, x, tau, out = None):
        ''' 
        Proximal operator of the convex conjugate of the function
        '''

        res = x.clone()

        jac_arr = self.vec.direct(res) # array of jacobian matrix  
        if self.order == 0:
            Z = make_hermitian_XTX(jac_arr)
        elif self.order == 1:
            Z = make_hermitian_XXT(jac_arr) # H = J J^T

        Z = regularise_matrices(Z, eps=1e-8) # Regularise matrices

        if Z.shape[1] == Z.shape[2] == 2:
            vals, vecs = eigen_2x2(Z)
        else:
            vals, vecs = numba_eigen_array(Z) # Find eigenvalues of H
        
        s = sqrt_eigenvals(vals) # Take square root of eigenvalues
        s_inv = pseudo_inverse(s) # Pseudo inverse of eigenvalues

        if self.norm == 'nuclear':
            prox_vals = prox_linf(s, tau) # Proximal operator of the nuclear norm
        elif self.norm == 'frobenius':
            prox_vals = prox_l2(s, tau) # Proximal operator of the frobenius norm

        if self.order == 0:
            prox_conj = apply_transform_XTX(jac_arr, vecs, prox_vals, s_inv)
        elif self.order == 1:
            prox_conj = apply_transform_XXT(jac_arr, vecs, prox_vals, s_inv)
                    
        if out is None:
            return self.vec.adjoint(prox_conj)
        else:
            out.fill(self.vec.adjoint(prox_conj))

    # 
    #@jit(forceobj=True)
    def proximal(self, x, tau, out = None):
        
        res = x.clone()

        Z = self.vec.direct(res) # array of jacobian matrix       

        if self.order == 0:
            h = make_hermitian_XTX(Z)
        elif self.order == 1:
            h = make_hermitian_XXT(Z) # H = J J^T

        Z = regularise_matrices(Z, eps=1e-8) # Regularise matrices

        if Z.shape[1] == Z.shape[2] == 2:
            vals, vecs = eigen_2x2(h)
        else:
            vals, vecs = numba_eigen_array(h) # Find eigenvalues of H
        
        s = sqrt_eigenvals(vals) # Take square root of eigenvalues

        s_inv = pseudo_inverse(s) # Pseudo inverse of eigenvalues
        
        if self.norm == 'nuclear':
            prox_vals = prox_l1(s, tau) # Proximal operator of the nuclear norm
        elif self.norm == 'frobenius':
            prox_vals = prox_l2(s, tau) # Proximal operator of the frobenius norm

        if self.order == 0:
            prox = apply_transform_XTX(Z, vecs, prox_vals, s_inv)
        elif self.order == 1:
            prox = apply_transform_XXT(Z, vecs, prox_vals, s_inv)
                    
        if out is None:
            return self.vec.adjoint(prox)
        else:
            out.fill(self.vec.adjoint(prox))

 
@njit(parallel=True)
def get_proximal_operator(U, s_prox, inv_S):
    out = np.zeros_like(U)
    l, n, _ = U.shape
    for i in prange(l):
        for j in prange(n):
            for k in prange(n):
                sum_inner = 0.0
                for q in prange(n):
                    sum_inner += s_prox[i, q] * inv_S[i, q] * U[i, q, k]
                for r in prange(n):
                    out[i, j, k] += U[i, j, r] * sum_inner
    return out


 
@njit(parallel=True)
def proximal_operation_XXT(A, X):
    l, n, _ = X.shape
    out = np.empty_like(X)
    for i in prange(l):
        for j in prange(n):
            for k in prange(n):
                sum_result = 0.0
                for r in prange(n):
                    sum_result += A[i, j, r] * X[i, r, k]
                out[i, j, k] = sum_result
    return out
 
@njit(parallel=True)
def proximal_operation_XTX(A, X):
    l, n, _ = X.shape
    out = np.empty_like(X)
    for i in prange(l):
        for j in prange(n):
            for k in prange(n):
                sum_result = 0.0
                for r in prange(n):
                    sum_result += X[i, j, r] * A[i, r, k]
                out[i, j, k] = sum_result
    return out

@njit(parallel=True)
def apply_transform_XXT(X, U, S, S_dagger):
    l, _, _ = X.shape  # assuming X is of shape (l, 3, 3)
    res = np.zeros_like(X)

    for i in prange(l):
        # U S S^dagger U^T X 
        US = np.dot(U[i], np.diag(S[i]))
        USS_dagger = np.dot(US, np.diag(S_dagger[i]))
        U_S_Sdagger_UT = np.dot(USS_dagger, np.transpose(U[i]))
        res[i] = np.dot(U_S_Sdagger_UT, X[i])
        
    return res

@njit(parallel=True)
def apply_transform_XTX(X, V, S, S_dagger):
    l, _, _ = X.shape  # assuming X is of shape (l, 3, 3)
    res = np.zeros_like(X)

    for i in prange(l):
        # X V S^dagger S V^T
        X_V = np.dot(X[i], V[i])
        X_V_S_dagger = np.dot(X_V, np.diag(S_dagger[i]))
        X_V_S_dagger_S = np.dot(X_V_S_dagger, np.diag(S[i]))
        res[i] = np.dot(X_V_S_dagger_S, np.transpose(V[i]))
        
    return res


##################
# Very Simple VTV
##################

@njit(parallel=True)
def svd(A):
    l, m, n = A.shape
    min_dim = np.minimum(m, n)
    
    U = np.zeros((l, m, min_dim))
    S = np.zeros((l, min_dim))
    V = np.zeros((l, min_dim, n))
    
    for i in prange(l):
        U[i], S[i], V[i] = np.linalg.svd(A[i], full_matrices=False)
        
    return U, S, V



@njit(parallel=True)
def recombine(U, S, Vt):
    l, n, m = U.shape[0], U.shape[1], Vt.shape[2]
    out = np.zeros((l, n, m))
    for i in prange(l):
        U_i = U[i]
        S_i = S[i]
        V_i = Vt[i]
        out_i = out[i]
        for j in prange(n):
            for k in prange(m):
                temp = 0.0
                for p in prange(S_i.shape[0]):
                    temp += U_i[j, p] * S_i[p] * V_i[p, k]
                out_i[j, k] = temp
    return out

@njit(parallel=True)
def max(X):
    l = X.shape[0]
    out = 0
    for i in prange(l):
        out = np.maximum(out, X[i])
    return out

@njit(parallel=True)
def l1(arr):
    l, n = arr.shape
    result = np.zeros(l)
    for i in prange(l):
        for j in range(n):
            result[i] += np.abs(arr[i, j])
    return result

@njit(parallel=True)
def l2(arr):
    l, n = arr.shape
    result = np.zeros(l)
    for i in prange(l):
        for j in range(n):
            result[i] += arr[i, j]**2
        result[i] = np.sqrt(result[i])
    return result

@njit(parallel=True)
def linf(arr):
    l, n = arr.shape
    result = np.zeros(l)
    for i in prange(l):
        max_val = abs(arr[i, 0])
        for j in range(1, n):  # Start from 1 since we've already considered the 0th element
            val = abs(arr[i, j])
            if val > max_val:
                max_val = val
        result[i] = max_val
    return result


@njit(parallel=True)
def prox_1(x, lam):
    """
    Proximal operator for the L1 norm
    """
    l, n = x.shape
    res = np.zeros((l, n))
    for i in prange(l):
        for j in range(n):
            tmp = np.abs(x[i, j]) - lam[i]
            res[i, j] = np.sign(x[i, j]) * np.maximum(0.0, tmp)
    return res

@njit(parallel=True)
def prox_2(x, lam):
    """
    Proximal operator for the L2 norm
    """
    l, n = x.shape
    res = np.zeros_like(x)
    for i in prange(l):
        norm_xi = np.linalg.norm(x[i, :])
        scaling_factor = np.maximum(0.0, 1.0 - lam[i] / norm_xi)
        res[i, :] = scaling_factor * x[i, :]
    return res

@njit(parallel=True)
def prox_inf(x, lam):
    """
    Proximal operator for the L-infinity norm
    """
    l, n = x.shape
    res = np.zeros_like(x)
    for i in prange(l):
        norm = np.sum(np.abs(x[i, :] / lam[i]))
        scaling_factor = 1.0 if norm <= 1 else 1.0 / norm
        res[i, :] = x[i, :] * scaling_factor - lam[i] * np.sign(x[i, :])
    return res



class SimpleVTV(Function):

    def __init__(self, domain_geometry, norm = 'nuclear'):

        super().__init__()
        self.nchannels = len(domain_geometry.containers)
        
        self.vec = BlockDataToVector(domain_geometry[0], nchannels=self.nchannels)

        self.norm = norm
        if self.norm != 'nuclear' and self.norm != 'frobenius':
            raise ValueError('norm must be nuclear or frobenius')

    def __call__(self, x):
        
        res = x.clone()
        arr = self.vec.direct(res)

        _, s, _ = svd(arr)

        s = np.maximum(s, 0)

        if self.norm == 'nuclear':
            res =  l1(s)
        else:
            res = l2(s)

        return np.sum(res)
    

    def convex_conjugate(self, x):

        res = x.clone()
        arr = self.vec.direct(res)

        _ , s, _ = svd(arr)

        s = np.maximum(s, 0)

        if self.norm == 'nuclear':
            res = linf(s)
        else:
            res = l2(s)

        return np.sum(res)
    
    def proximal(self, x, tau, out=None):

        res = x.clone()
        arr = self.vec.direct(res)

        u, s, v_t = svd(arr)

        s = np.maximum(s, 0)

        if isinstance(tau, Number):
            t = tau * np.ones(arr.shape[0])
        else:
            t = tau.as_array().flatten()

        if self.norm == 'nuclear':
            s_prox = prox_1(s, t)
        else:
            s_prox = prox_2(s, t)

        prox = recombine(u, s_prox, v_t)

        res = self.vec.adjoint(prox)

        if out is None:
            return res
        else:
            out.fill(res)
            
     
    def proximal_conjugate(self, x, tau, out=None):
        res = x.clone()

        arr = self.vec.direct(res)

        u, s, v_t = svd(arr)

        s = np.maximum(s, 0)

        if isinstance(tau, Number):
            t = tau * np.ones(arr.shape[0])
        else:
            t = tau.as_array().flatten()

        if self.norm == 'nuclear':
            s_prox = prox_inf(s, t)
        else:
            s_prox = prox_2(s, t)

        prox = recombine(u, s_prox, v_t)

        if out is None:
            return self.vec.adjoint(prox)
        else:
            out.fill(self.vec.adjoint(prox))



        





