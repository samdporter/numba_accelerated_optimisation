from cil.optimisation.functions import Function
import numpy as np

def rmul(self, array):
    """ 
    Multiply the function by the array. 
    """
    return ArrayScaledFunction(self, array)

def mul(self, array):
    """ 
    Multiply the function by the array. 
    """
    return ArrayScaledFunction(self, array)

Function.__rmul__ = rmul
Function.__mul__ = mul

# Path: src/cil_extensions/ArrayScaledFunction.py

class ArrayScaledFunction(Function):

    def __init__(self, function, scalar):
        
        super(ArrayScaledFunction, self).__init__() 
                                                     
        if not isinstance (scalar, np.array):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        
        self.scalar = scalar
        self.function = function       
    @property
    def L(self):
        if self._L is None:
            if self.function.L is not None:
                self._L = abs(self.scalar) * self.function.L
            else:
                self._L = None
        return self._L
    @L.setter
    def L(self, value):
        # call base class setter
        super(ScaledFunction, self.__class__).L.fset(self, value )

    @property
    def scalar(self):
        return self._scalar
    @scalar.setter
    def scalar(self, value):
        if isinstance(value, (Number, )):
            self._scalar = value
        else:
            raise TypeError('Expecting scalar type as a number type. Got {}'.format(type(value)))
    def __call__(self,x, out=None):
        r"""Returns the value of the scaled function.
        
        .. math:: G(x) = \alpha F(x)
        
        """
        return self.scalar * self.function(x)

    def convex_conjugate(self, x):
        r"""Returns the convex conjugate of the scaled function.
        
        .. math:: G^{*}(x^{*}) = \alpha  F^{*}(\frac{x^{*}}{\alpha})
        
        """
        try:
            x.divide(self.scalar, out = x)
            tmp = x
        except TypeError:
            tmp = x.divide(self.scalar, dtype=np.float32)

        val = self.function.convex_conjugate(tmp)

        if id(tmp) == id(x):
            x.multiply(self.scalar, out = x)

        return  self.scalar * val

    
    def gradient(self, x, out=None):
        r"""Returns the gradient of the scaled function.
        
        .. math:: G'(x) = \alpha  F'(x)
        
        """
        if out is None:            
            return self.scalar * self.function.gradient(x)
        else:
            self.function.gradient(x, out=out)
            out *= self.scalar  

    def proximal(self, x, tau, out=None):
        
        r"""Returns the proximal operator of the scaled function.
        
        .. math:: \mathrm{prox}_{\tau G}(x) = \mathrm{prox}_{(\tau\alpha) F}(x)
        
        """        

        return self.function.proximal(x, tau*self.scalar, out=out)     


    def proximal_conjugate(self, x, tau, out = None):
        r"""This returns the proximal operator for the function at x, tau
        """
        try:
            tmp = x
            x.divide(tau, out = tmp)
        except TypeError:
            tmp = x.divide(tau, dtype=np.float32)

        if isinstance(tau, Number):
            if out is None:
                val = self.function.proximal(tmp, self.scalar/tau )
            else:
                self.function.proximal(tmp, self.scalar/tau, out = out)
                val = out     
        else:
            scalar = tau.clone()
            scalar.fill(self.scalar)
            if out is None:
                val = self.function.proximal(tmp, scalar/tau)
            else:
                self.function.proximal(tmp, scalar/tau, out = out)
                val = out

        if id(tmp) == id(x):
            x.multiply(tau, out = x)

        # CIL issue #1078, cannot use axpby
        #val.axpby(-tau, 1.0, x, out=val)
        val.multiply(-tau, out = val)
        val.add(x, out = val)

        if out is None:
            return val