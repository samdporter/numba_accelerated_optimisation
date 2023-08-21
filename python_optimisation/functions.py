import numpy as np
from numba import njit, prange
from numbers import Number
from python_optimisation.misc import multiply_array

######################
## Function Class ####
######################

class Function():

    def __init__(*args, **kwargs):
        return

    def __call__():
        raise NotImplementedError

    def convex_conjugate(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError

    def proximal(self, x):
        raise NotImplementedError

    def proximal_conjugate(self, x, tau):
        return x - tau * self.proximal(x / tau, 1 / tau)
    
    def multiply(self, other):
        if isinstance(other, Number):
            return ScalarScaledFunction(self, other)
        elif isinstance(other, np.ndarray):
            return ArrayScaledFunction(self, other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)
    
class ArrayScaledFunction():
    
    def __init__(self, function, weight):
        self.function = function
        self.weight = weight
        
    def __call__(self, x):
        try:
            return np.sum(self.weight * self.function.call_no_sum(x))
        except:
            return np.sum(self.weight * self.function(x))

    def convex_conjugate(self, x):
        # definitely not correct
        return np.sum(self.weight) * self.function.convex_conjugate(self.divide(x,self.weight))

    def gradient(self, x):
        
        return multiply_array(self.function.gradient(x), self.weight)

    def proximal(self, x, tau):
        return self.function.proximal(x, self.weight*tau)

    def proximal_conjugate(self, x, tau):
        
        tmp = self.divide(x, tau)
        if isinstance(tau, Number):
            val = self.function.proximal(tmp, self.weight / tau)
        else:
            val = self.function.proximal(tmp, self.divide(self.weight, tau))
        return x - tau*val
    
    def multiply(self, other):
        return ArrayScaledFunction(self.function, other*self.weight)
    
    ## overloading operations
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)

    @staticmethod
    @njit
    def divide(x,y,eps = 10e-10):
        return np.divide(x,y+eps)
    
class ScalarScaledFunction():
    
    def __init__(self, function, scalar):
        self.function = function
        self.scalar = scalar
        
    def __call__(self, x):
        return self.scalar * self.function(x) 

    def convex_conjugate(self, x):
        return self.scalar * self.function.convex_conjugate(x/self.scalar)

    def gradient(self, x):
        
        return self.scalar * self.function.gradient(x)

    def proximal(self, x, tau):
        return self.function.proximal(x, self.scalar*tau)

    def proximal_conjugate(self, x, tau):
        
        tmp = self.divide(x, tau)
        if isinstance(tau, Number):
            val = self.function.proximal(tmp, self.scalar / tau)
        else:
            val = self.function.proximal(tmp, self.divide(self.scalar, tau))
        return x - tau*val
    
    def multiply(self, other):
        if isinstance(other, Number):
            return ScalarScaledFunction(self.function, self.scalar*other)
        elif isinstance(other, np.ndarray):
            return ArrayScaledFunction(self.function, self.scalar*other)

    ## overloading operations
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        return self.multiply(other)

    @staticmethod
    @njit
    def divide(x,y,eps = 10e-10):
        return np.divide(x,y+eps)
    
class BlockFunction(Function):

    def __init__(self, functions,*args, **kwargs):
        
        self.functions = functions

        try:
            self.L = np.max([function.L for function in self.functions])
        except:
            self.L = None

    def __call__(self, x):
        out = 0
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out += function(arr)
        return out

    def convex_conjugate(self, x):
        out = 0
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out += function.convex_conjugate(arr)
        return out

    def gradient(self, x):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out.append(function.gradient(arr))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res

    def proximal(self, x, tau):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            out.append(function.proximal(arr, tau))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res

    def proximal_conjugate(self, x, tau):
        out = []
        # put last dimension first
        x = np.moveaxis(x, -1, 0)
        for function, arr in zip(self.functions, x):
            try:
                out.append(function.proximal_conjugate(arr, tau))
            except:
                out.append(arr - tau * function.proximal(arr / tau, 1 / tau))
        res = np.array(out)
        # put first dimension last
        res = np.moveaxis(res, 0, -1)
        return res
    
    ## overloading operations - can't do these yet
    def __mul__(self, other):
        raise NotImplementedError
    
    def __rmul__(self, other):
        raise NotImplementedError
    
class SumFunction(Function):

    def __init__(self, functions,*args, **kwargs):
        self.functions = functions

        try:
            self.L = np.max([function.L for function in self.functions])
        except:
            self.L = None
            
    def __call__(self, x):
        out = 0
        for function in self.functions:
            out += function(x)
        return out
    
    def convex_conjugate(self, x):
        out = 0
        for function in self.functions:
            out += function.convex_conjugate(x)
        return out
    
    def gradient(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.gradient(x)
        return out
    
    def proximal(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.proximal(x)
        return out
    
    def proximal_conjugate(self, x):
        
        out = np.zeros_like(x)
        for function in self.functions:
            out += function.proximal_conjugate(x)
        return out
    
class OperatorCompositionFunction(Function):
    
    def __init__(self, function, operator):
        self.function = function
        self.operator = operator
        
    def __call__(self, x):
        return self.function(self.operator.direct(x))
    
    def convex_conjugate(self, x):
        return self.function.convex_conjugate(self.operator.direct(x))
    
    def gradient(self, x):
        return self.operator.adjoint(self.function.gradient(self.operator.direct(x)))
    
    def proximal(self, x):
        return self.operator.adjoint(self.function.proximal(self.operator.direct(x)))
    
    def proximal_conjugate(self, x, tau):
        return self.operator.adjoint(self.function.proximal_conjugate(self.operator.direct(x), tau))

class IdentityFunction(Function):

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x

    def gradient(self, x):
        return np.ones_like(x)

    def hessian(self, x):
        return np.zeros_like(x)
    
    def proximal(self, x):
        return x

class ZeroFunction(Function):

    def __init__(self) -> None:
        self.L = 0

    def __call__(self, x):
        return 0
    
    def gradient(self, x):
        return np.zeros_like(x)
    
    def hessian(self, x):
        return np.zeros_like(x)
    
    def proximal(self, x):
        return np.zeros_like(x)
    
######################
## Least Squares #####
######################

@njit(parallel=True)
def _ls_call(y, x, w):
    out = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                diff = y[i, j, k] - x[i, j, k]
                out += 0.5 * w[i,j,k] * diff**2
    return out

@njit(parallel=True)
def _ls_grad(y, x, w):
    out = np.zeros_like(y)
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                out[i, j, k] = w[i,j,k] * (x[i, j, k] - y[i, j, k])
    return out

@njit(parallel=True)
def _ls_prox(y, x, tau, w):
    out = np.zeros_like(y)
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                out[i, j, k] = (y[i, j, k] - x[i, j, k]) / (1 + w[i,j,k] * tau) + x[i, j, k]
    return out

@njit(parallel=True)
def _ls_convex_conjugate(y, b, w):
    ''' 
    written by chat-gpt
    Should check
    '''
    out = 0
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                term = (y[i, j, k] + w[i, j, k] * b[i, j, k]) / (2 * w[i, j, k])
                out += term * y[i, j, k] - 0.5 * w[i, j, k] * (term - b[i, j, k])**2
    return out


class L2NormSquared(Function):

    def __init__(self, b, weights = None) -> None:

        self.b = b

        if weights is None:
            self.weights = np.ones_like(b)
        else:
            self.weights = weights
    
        self.L = 1 * np.max(self.weights)

    def __call__(self, x):
        return _ls_call(self.b, x, self.weights)

    def convex_conjugate(self, x):
        # Keeping the definition as 0.5 times the squared L2 norm
        return 0.5 * _ls_call(self.b, x, self.weights)
     
    def gradient(self, x):
        return _ls_grad(self.b, x, self.weights)
     
    def proximal(self, x, tau):
        return _ls_prox(self.b, x, tau, self.weights)
    
######################
## L21 Norm ##########
######################

@njit(parallel=True)
def _l21_norm(y):
    out = 0
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                sum_sq = 0.
                for c in prange(y.shape[3]):
                    sum_sq += y[i, j, k, c] ** 2
                out += np.sqrt(sum_sq)
    return out

@njit(parallel=True)
def _l21_prox(y, tau):
    out = np.zeros_like(y)
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                sum_sq = 0.
                for c in prange(y.shape[3]):
                    sum_sq += y[i, j, k, c] ** 2
                norm = np.sqrt(sum_sq)
                if norm == 0:
                    continue
                a = norm / tau
                el = np.maximum(a - 1, 0)
                m = el / a
                for c in prange(y.shape[3]):
                    out[i, j, k, c] = m * y[i, j, k, c]
    return out


class L21Norm(Function):

    def __init__(self):
        # The Lipschitz constant for L21 norm is not trivial to compute. 
        # It can depend on the specifics of the problem. 
        # Here it's initialized to a placeholder and should be set appropriately.
        self.L = 1  # Placeholder

    def __call__(self, x):
        return _l21_norm(x)

    def convex_conjugate(self, x):
        # The convex conjugate of the L21 norm is not trivial to express.
        # This would be a placeholder or could raise NotImplementedError.
        raise NotImplementedError("Convex conjugate of L21 norm is not implemented.")

    def gradient(self, x):
        # The L21 norm is non-differentiable. 
        # You'd typically use a proximal method to handle this.
        raise NotImplementedError("Gradient of L21 norm is not defined.")

    def proximal(self, x, tau):

        return _l21_prox(x, tau)
        

######################
## smooth L21 Norm ###
######################

@njit(parallel=True)
def _smoothed_l21_norm(y, epsilon):
    out = 0.0
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                sum_sq = 0.
                for c in prange(y.shape[3]):
                    sum_sq += y[i, j, k, c] ** 2
                norm = np.sqrt(sum_sq + epsilon**2)
                out += norm
    return out


@njit(parallel=True)
def _smoothed_l21_gradient(y, epsilon):
    out = np.zeros_like(y)
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                sum_sq = 0.
                for c in prange(y.shape[3]):
                    sum_sq += y[i, j, k, c] ** 2
                norm = np.sqrt(sum_sq + epsilon**2)
                for c in prange(y.shape[3]):
                    out[i, j, k, c] = y[i, j, k, c] / norm
    return out


class SmoothedL21Norm(Function):

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        self.L = 1  # Placeholder

    def __call__(self, x):
        return _smoothed_l21_norm(x, self.epsilon)

    def convex_conjugate(self, x):
        # Placeholder
        raise NotImplementedError("Convex conjugate of smoothed L21 norm is not implemented.")

    def gradient(self, x):
        return _smoothed_l21_gradient(x, self.epsilon)

    def proximal(self, x, tau):
        # Using the original proximal as it's the same
        return _l21_prox(x, tau)



######################
## JTV ###############
######################

# TODO: guided JTV should be done with operator

@njit(parallel=True)
def _jtv_call(x, y):
    out = 0
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out += np.sqrt(x[d, h, w] ** 2 + y[d, h, w] ** 2)
    return out

@njit(parallel=True)
def _jtv_convex_conjugate(x, y):
    ''' should be inidicator '''
    raise NotImplementedError

@njit(parallel=True)
def _jtv_grad(x, y, e):
    out_x = np.zeros_like(x)
    out_y = np.zeros_like(y)
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out_x[d, h, w] = x[d, h, w] / np.sqrt(x[d, h, w] ** 2 + y[d, h, w] ** 2 + e ** 2)
                out_y[d, h, w] = y[d, h, w] / np.sqrt(x[d, h, w] ** 2 + y[d, h, w] ** 2 + e ** 2)
    return out_x, out_y

@njit(parallel=True)
def _jtv_prox(x, y, tau):
    out_x = np.zeros_like(x)
    out_y = np.zeros_like(y)
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                norm = np.sqrt(x[d, h, w] ** 2 + y[d, h, w] ** 2)
                tmp = norm - tau
                if tmp > 0:
                    out_x[d, h, w] = tmp * x[d, h, w] / norm
                    out_y[d, h, w] = tmp * y[d, h, w] / norm
    return out_x, out_y

@njit(parallel=True)
def _jtv_call_no_sum(x, y, e):
    out = 0
    for d in prange(x.shape[0]):
        for h in prange(x.shape[1]):
            for w in prange(x.shape[2]):
                out += np.sqrt(x[d, h, w] ** 2 + y[d, h, w] ** 2 + e ** 2)
    return out

class JTV(Function):

    def __init__(self,epsilon=None):
        self.epsilon = epsilon

    def __call__(self, x, y):
        return _jtv_call(x, y)
    
    def convex_conjugate(self, x, y):
        if self.epsilon is None:
            raise NotImplementedError
        else:
            return _jtv_convex_conjugate(x, y)

    def gradient(self, x, y):
        if self.epsilon is None:
            raise NotImplementedError
        else:
            return _jtv_grad(x, y, self.epsilon)
        
    def proximal(self, x, y, tau):
        return _jtv_prox(x, y, tau)
    
    # annoyingly need this for ArrayScaledFunction
    def call_no_sum(self, x, y):
        if self.epsilon is None:
            return _jtv_call_no_sum(x, y, 0)
        else:
            return _jtv_call_no_sum(x, y, self.epsilon)
        
##################
# Very Simple VTV
##################

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

@njit
def project_l1_ball(v):
    """Project a vector onto the unit l1 ball."""
    u = np.abs(v)
    if np.sum(u) <= 1:
        return v

    n = len(v)
    s = np.sort(u)[::-1]
    tmpsum = 0.0

    for i in range(n - 1):
        tmpsum += s[i]
        t = (tmpsum - 1) / (i + 1)
        if t >= s[i + 1]:
            return np.sign(v) * np.maximum(u - t, 0)

    return np.sign(v) * np.maximum(u - s[-1], 0)

@njit(parallel=True)
def prox_inf(x, lam):
    """
    Proximal operator for the L-infinity norma.
    """
    l, n = x.shape
    out = np.empty_like(x)
    for i in prange(l):
        scaled_x = x[i] / lam[i]
        proj = project_l1_ball(scaled_x)
        out[i] =  x[i] - lam[i] * proj
    return out


def flatten_dims(array):
    """
    Reshape the input array into a 2D array, while keeping the last two dimensions intact.
    """
    if array.ndim <= 3:
        raise ValueError("The input array must have more than 3 dimensions.")
    
    new_shape = (np.prod(array.shape[:-2]),) + array.shape[-2:]
    return array.reshape(new_shape)

def restore_dims(array, original_shape):
    """
    Restore the dimensions of the input 2D array to match the provided original shape.
    """
    return array.reshape(original_shape)


class VectorialTotalVariation(Function):

    def __init__(self, norm = 'nuclear'):

        super().__init__()

        self.norm = norm

    def __call__(self, x):

        x = flatten_dims(x)

        _, s, _ = svd(x)

        s = np.maximum(s, 0)

        if self.norm == 'nuclear':
            res =  l1(s)
        else:
            res = l2(s)

        return np.sum(res)

    def convex_conjugate(self, x):

        x = flatten_dims(x)

        _ , s, _ = svd(x)

        s = np.maximum(s, 0)

        if self.norm == 'nuclear':
            res = linf(s)
        else:
            res = l2(s)

        return np.sum(res)
    
    def proximal(self, x, tau):

        shape = x.shape
        x = flatten_dims(x)

        u, s, v_t = svd(x)

        s = np.maximum(s, 0)

        if isinstance(tau, Number):
            t = tau * np.ones(x.shape[0])
        else:
            t = tau.flatten()

        if self.norm == 'nuclear':
            s_prox = prox_1(s, t)
        else:
            s_prox = prox_2(s, t)

        prox = recombine(u, s_prox, v_t)

        return restore_dims(prox, shape)
            
    def proximal_conjugate(self, x, tau):

        return super().proximal_conjugate(x, tau)

        shape = x.shape
        x = flatten_dims(x)

        u, s, v_t = svd(x)

        s = np.maximum(s, 0)

        if isinstance(tau, Number):
            t = tau * np.ones(x.shape[0])
        else:
            t = tau.flatten()

        if self.norm == 'nuclear':
            s_prox = prox_inf(s, t)
        else:
            s_prox = prox_2(s, t)

        prox_conj = recombine(u, s_prox, v_t)

        return restore_dims(prox_conj, shape)
    
######################
## Indicator #########
######################

@njit
def _indic_call(x, l, u):
    for i in prange(x.size):
        if x.flat[i] <l:
            return np.inf
        elif x.flat[i] > u:
            return np.inf
    return 0

@njit(parallel=True)
def _indic_prox(x, l, u, tau):
    out = np.zeros_like(x)
    for i in prange(len(x)):
        out[i] = np.clip(x[i], l, u)
    return out

class Indicator(Function):

    def __init__(self, lower=-np.inf, upper = np.inf):

        super().__init__()

        self.lower = lower
        self.upper = upper

     
    def __call__(self, x):
        return _indic_call(x, self.lower, self.upper)
    
     
    def proximal(self, x, tau):
        return _indic_prox(x, self.lower, self.upper, tau)
    
######################
## Kullback Leibler ##
######################


@njit(parallel=True)
def _kl_call(b, x, eta):
    out = 0
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                val = x[i, j, k] + eta
                out += b[i, j, k] * np.log((b[i, j, k] + eta) \
                                    / (val)) - b[i, j, k] + val
    return out

@njit(parallel=True)
def kl_convex_conjugate(b, x, eta):
    accumulator = 0.
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                y = 1 - x[i, j, k]
                if y > 0:
                    if x[i, j, k] > 0:
                        accumulator += x[i, j, k] * np.log(y)
                    accumulator += eta[i, j, k] * x[i, j, k]
    return - accumulator


@njit(parallel=True)
def _kl_grad(b, x, eta):
    grad = np.zeros_like(b)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                grad[i, j, k] = 1 - (b[i, j, k]) / (x[i, j, k] + eta)
    return grad

@njit(parallel=True)
def kl_proximal(b, x, tau, eta):
    out = np.zeros_like(x)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                under_root = (x[i, j, k] + eta - tau)**2 + 4 * tau * b[i, j, k]
                out[i, j, k] = ( x[i, j, k] - eta - tau ) + np.sqrt( under_root)
                out[i, j ,k] /= 2
    return out

@njit(parallel=True)
def kl_proximal_conjugate(x, b, eta, tau):
    out = np.zeros_like(x)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                z = x[i, j, k] + ( tau * eta)
                out[i, j, k] = 0.5 * (
                    (z + 1) - np.sqrt((z-1)*(z-1) + 4 * tau * b[i, j, k])
                )
    return out

class KullbackLeibler:

    def __init__(self, b, eta=0):
        self.b = b
        
        self.eta = eta

        super().__init__()

    def __call__(self, x):
        return _kl_call(self.b, x, self.eta)

    def gradient(self, x):
        if self.eta == 0:
            raise NotImplementedError("Gradient of Kullback-Leibler divergence is not defined for eta = 0.")
        return _kl_grad(self.b, x, self.eta)

    def proximal(self, x, tau):
        return kl_proximal(self.b, x, tau, self.eta)

    def convex_conjugate(self, x):
        return kl_convex_conjugate(self.b, x, self.eta)
    
    def proximal_conjugate(self, x, tau):
        return kl_proximal_conjugate(self.b, x, self.eta, tau)
    
@njit(parallel=True)
def _weighted_kl_call(b, x, eta, weights):
    out = 0
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                val = x[i, j, k] + eta[i, j, k]
                out += weights[i, j, k] * (b[i, j, k] * np.log((b[i, j, k] + eta) / (val + eta)) - b[i, j, k] + val)
    return out

@njit(parallel=True)
def _weighted_kl_grad(b, x, eta, weights):
    grad = np.zeros_like(b)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                grad[i, j, k] = weights[i, j, k] * (1 - (b[i, j, k] + eta) / (x[i, j, k] + eta))
    return grad

class WeightedKullbackLeibler:

    def __init__(self, b, eta=None, weights=None):
        self.b = b
        if eta is None:
            self.eta = 1e-15
        else:
            self.eta = eta
        if weights is None:
            self.weights = np.ones_like(b)
        else:
            self.weights = weights

        super().__init__()

    def __call__(self, x):
        return _weighted_kl_call(self.b, x, self.eta, self.weights)

    def gradient(self, x):
        return _weighted_kl_grad(self.b, x, self.eta, self.weights)

    def proximal(self, x, tau):
        # Using gradient descent for the proximal operator
        return x - tau * self.gradient(x)

    def proximal_conjugate(self, x, tau):
        return x - tau * self.proximal(x / tau, 1/tau)

    def convex_conjugate(self, x):
        # The convex conjugate of the Kullback-Leibler divergence is complex 
        # Here, we provide an approximation based on its definition
        return _weighted_kl_call(self.b, -np.log(1 - x), self.eta, self.weights)
    
######################
## MAPEM function ####
######################

@njit(parallel=True)
def _mapem_call(x, s, lam, eps = 1e-10):

    acc = 0. 
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                val = x[i, j, k] + eps
                tmp = val - lam[i, j, k] * np.log(val)
                acc += tmp * s[i, j, k]
    return acc

def _mapem_call_zeroed(x,s,lam,eps=1e-10):
    
    acc = 0.
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                val = x[i, j, k] + eps
                out = lam[i, j, k] * np.log((lam[i, j, k] + eps) \
                                    / (val)) - lam[i, j, k] + val
                out *= s[i, j, k]
                acc += out
    return acc

@njit(parallel=True)
def _mapem_convex_conjugate(x, s, lam):
    acc = 0.
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                if x[i, j, k] > s[i, j, k]:
                    return np.inf
                if s[i, j, k] == 0:
                    continue
                sl = s[i, j, k] * lam[i, j, k]
                var = np.log(s[i, j, k] - x[i, j, k])
                const = np.log(sl) - 1
                acc += (const - var) * sl
    
    return acc

@njit(parallel=True)
def _mapem_gradient(x, s, lam, eps):
    result = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                result[i, j, k] = s[i, j, k] * (1 - lam[i, j, k] / (x[i, j, k] + eps))

    return result

@njit(parallel=True)
def _mapem_proximal(x, tau, s, lam):
    result = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                under_root = (x[i, j, k] - tau * s[i, j, k])**2 + 4 * tau * lam[i, j, k] * s[i, j, k]
                result[i, j, k] = (x[i, j, k] - tau * s[i, j, k] + np.sqrt(under_root))
                result[i, j, k] /= 2

    return result
    
class MAPEM(Function):
    """ 
    function for use in the MAP-EM algorithm
    """

    def __init__(self, lam, s, **kwargs):

        super().__init__()

        self.lam = lam
        self.s = s
        self.eps=1e-8

    def __call__(self, x):

        return _mapem_call_zeroed(x, self.s, self.lam, self.eps)
    
    def proximal(self, x, tau):

        return _mapem_proximal(x, tau, self.s, self.lam)

    def convex_conjugate(self, x):

        return _mapem_convex_conjugate(x, self.s, self.lam)

    def gradient(self, x):

        return _mapem_gradient(x, self.s, self.lam, self.eps)
    
######################
## Quadratic prior ###
######################

@njit(parallel=True)
def _quad_call(x, size=3):
    out = 0
    shape = x.shape
    offset = size // 2
    for d in prange(offset, shape[0] - offset):
        for h in prange(offset, shape[1] - offset):
            for w in prange(offset, shape[2] - offset):
                center_val = x[d, h, w]
                # Iterate over the neighborhood
                for dd in prange(-offset, offset+1):
                    for hh in prange(-offset, offset+1):
                        for ww in prange(-offset, offset+1):
                            out += (center_val - x[d+dd, h+hh, w+ww])**2
    return out

@njit(parallel=True)
def _quad_grad(x, size=3):
    grad = np.zeros_like(x)
    shape = x.shape
    offset = size // 2
    for d in prange(offset, shape[0] - offset):
        for h in prange(offset, shape[1] - offset):
            for w in prange(offset, shape[2] - offset):
                center_val = x[d, h, w]
                for dd in prange(-offset, offset+1):
                    for hh in prange(-offset, offset+1):
                        for ww in prange(-offset, offset+1):
                            grad[d, h, w] += 2 * (center_val - x[d+dd, h+hh, w+ww])
    return grad

@njit(parallel=True)
def _quad_prox(x, tau, size=3):
    out = np.zeros_like(x)
    shape = x.shape
    offset = size // 2
    neighbors_count = size ** 3 - 1
    for d in prange(offset, shape[0] - offset):
        for h in prange(offset, shape[1] - offset):
            for w in prange(offset, shape[2] - offset):
                neighbors_sum = 0
                for dd in prange(-offset, offset+1):
                    for hh in prange(-offset, offset+1):
                        for ww in prange(-offset, offset+1):
                            neighbors_sum += x[d+dd, h+hh, w+ww]
                out[d, h, w] = (x[d, h, w] + tau * neighbors_sum) / (1 + neighbors_count * tau)
    return out

class QuadraticPrior(Function):

    def __init__(self, size=3):
        self.size = size

    def __call__(self, x):
        return _quad_call(x, self.size)

    def gradient(self, x):
        return _quad_grad(x, self.size)

    def proximal(self, x, tau):
        return _quad_prox(x, tau, self.size)

    def convex_conjugate(self, x):
        pass






    