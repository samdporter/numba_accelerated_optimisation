import numpy as np
from numba import jit, njit, prange

from numbers import Number

from misc.misc import timer

######################################
###### Numba accelerated functions ###
######################################

# leftovers after refactoring

######################
## eigenvals #########
######################
                          
# not working atm - only does a small portion of the array
@njit(parallel=True)
def eigen_array(J):
    """ Calculate the eigenvalues and eigenvectors of a 3D array of matrices """
    val = np.zeros(J.shape[:-1])
    vec = np.zeros(J.shape)

    for i in prange(J.shape[0]):
        for j in prange(J.shape[1]):
            for k in prange(J.shape[2]):
                eig_val, eig_vec = np.linalg.eig(J[i, j, k])
                val[i, j, k] = eig_val
                vec[i, j, k] = eig_vec

    return val, vec
         
        

######################################
###### Function Classes ##############
#####################################
    
######################
## JgTV ##############
######################

@njit(parallel=True)
def _jgtv_call(x, a):
    out = 0
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out += np.sqrt(np.dot(np.transpose(x[d, h, w]),x[d, h, w])  +np.dot(np.transpose(a[d, h, w]),a[d, h, w]))
    return out

@njit(parallel=True)
def _jgtv_convex_conjugate(x, a):
    ''' should be inidicator '''
    raise NotImplementedError

@njit(parallel=True)
def _jgtv_grad(x, a, e):
    out = np.zeros_like(x)
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out[d, h, w] = x[d, h, w] / np.sqrt(np.dot(np.transpose(x[d, h, w]),x[d, h, w])  \
                    +np.dot(np.transpose(a[d, h, w]),a[d, h, w]) + e ** 2)
    return out

@njit(parallel=True)
def _jgtv_prox(x, a, tau):
    out = np.zeros_like(x)
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                norm = np.sqrt(np.dot(np.transpose(x[d, h, w]),x[d, h, w]) + np.dot(np.transpose(a[d, h, w]),a[d, h, w]))
                tmp = norm - tau
                if tmp > 0:
                    out[d, h, w] = tmp * x[d, h, w] / norm
    return out

@njit(parallel=True)
def _jgtv_call_no_sum(x, a, e):
    out = np.zeros_like(x)
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out[d,h,w] = np.sqrt(np.dot(np.transpose(x[d, h, w]),x[d, h, w])  \
                    +np.dot(np.transpose(a[d, h, w]),a[d, h, w]) + e ** 2)
    return out


class JgTV(Function):
    
    """ tau must now be an array """

    def __init__(self, anatomical, epsilon=None):
        self.anatomical = anatomical
        self.epsilon = epsilon

    def __call__(self, x):
        return _jgtv_call(x, self.anatomical)
    
    def convex_conjugate(self, x):
        if self.epsilon is None:
            raise NotImplementedError
        else:
            return _jgtv_convex_conjugate(x, self.anatomical)

    def gradient(self, x):
        if self.epsilon is None:
            raise NotImplementedError
        else:
            return _jgtv_grad(x, self.anatomical, self.epsilon)
       
    def proximal(self, x, tau):
        return _jgtv_prox(x, self.anatomical, tau)  

    # annoyingly need this for ArrayScaledFunction
    def call_no_sum(self, x):
        if self.epsilon is None:
            return _jgtv_call_no_sum(x, self.anatomical, 0)
        else:
            return _jgtv_call_no_sum(x, self.anatomical, self.epsilon)  


    



