from cil.optimisation.functions import Function

class MAPEM(Function):
    """ 
    function for use in the MAP-EM algorithm
    """

    def __init__(self, lam, s, **kwargs):
        super(MAPEM, self).__init__(**kwargs)

        self.lam = lam
        self.s = s
        self.one = s.get_uniform_copy(1.)
        self.eps=1e-8
        self.additive = 1e7-4e6

    def __call__(self, x):
        return _call(x.as_array()+self.eps, self.s.as_array(), self.lam.as_array())
    
        tmp = x - self.lam * (x+self.eps).log()
        return tmp.dot(self.s)
    
    def proximal(self, x, tau, out=None):
        under_root = (x - tau * self.s).power(2) + (4*tau*self.s).multiply(self.lam)
        if out is None:
            res = (x - tau * self.s + under_root.sqrt())
            return res/2
        
        else:
            x.subtract(tau*self.s, out=out)
            under_root.sqrt(out=out)
            out.add(x, out=out)
            out.divide(2, out=out)

    def convex_conjugate(self, p):

        return _convex_conjugate(p.as_array(), self.s.as_array(), self.lam.as_array())

        var = (self.s - p).log()
        const = (self.s.multiply(self.lam)).log() - self.one
        
        # C - <log(s - x), s*lam>
        return const.dot(self.s*self.lam) - var.dot(self.s*self.lam)

    def gradient(self, x, out=None):

        if out is None:
            return self.s - self.s.multiply(self.divide(self.lam, x))
        
        else:
            out.fill(self.s - self.s.multiply(self.divide(self.lam, x)))
    
    @staticmethod
    def divide(a, b, eps=1e-8):
        return a.divide(b+eps, out=None)
    
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def _call(x, s, lam):
    n = x.size
    acc = np.zeros(n) # accumulator array

    for i in prange(n):
        if s.flat[i] != 0:
            tmp = x.flat[i] - lam.flat[i] * np.log(x.flat[i])
            acc[i] = tmp * s.flat[i]

    return np.sum(acc)

@njit(parallel=True)
def _convex_conjugate(p, s, lam):
    n = s.size
    result = np.zeros(n)
    for i in prange(n):
        if p.flat[i] > s.flat[i]:
            return np.inf
        if s.flat[i] == 0:
            result[i] = 0
            continue
        var = np.log(s.flat[i] - p.flat[i])
        const = np.log(s.flat[i]*lam.flat[i]) - 1
        result[i] = (const - var) * s.flat[i] * lam.flat[i]
    return np.sum(result)
