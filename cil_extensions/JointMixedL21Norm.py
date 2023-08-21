# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.functions import Function
from cil.framework import BlockDataContainer
import numpy as np
from numbers import Number

from misc.misc import timer
import time



has_numba = True
try:
    import numba
    @numba.jit(parallel=True, nopython=True)
    def _proximal_step_numba(arr, abstau):
        '''Numba implementation of a step in the calculation of the proximal of MixedL21Norm
        
        Parameters:
        -----------
        arr : numpy array, best if contiguous memory. 
        abstau: float >= 0
        Returns:
        --------
        Stores the output in the input array.
        Note:
        -----
        
        Input arr should be contiguous for best performance'''
        tmp = arr.ravel()
        for i in numba.prange(tmp.size):
            if tmp[i] == 0:
                continue
            a = tmp[i] / abstau
            el = a - 1
            if el <= 0.0:
                el = 0.
            
            tmp[i] = el / a 
        return 0
except ImportError:
    has_numba = False
    


def _proximal_step_numpy(tmp, tau):
    '''Numpy implementation of a step in the calculation of the proximal of MixedL21Norm
    
    Parameters:
    -----------
    tmp : DataContainer/ numpy array, best if contiguous memory. 
    tau: float or DataContainer
    Returns:
    --------
    A DataContainer where we have substituted nan with 0.
    '''
    # Note: we divide x by tau so the cases of tau both scalar and 
    # DataContainers run
    tmp /= np.abs(tau, dtype=np.float32)
    res = tmp - 1
    res.maximum(0.0, out=res)
    res /= tmp

    resarray = res.as_array()
    resarray[np.isnan(resarray)] = 0
    res.fill(resarray)
    return res

class MixedL21Norm(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 
        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        res = x[0].containers[0].get_uniform_copy(0)
        for el in x.containers:
            res += el.pnorm(2).power(2)
        return res.sqrt().sum()                              
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """

        # def wrong

        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        res = 0

        tmp = x.containers[0].copy().fill(0)

        for el in x.containers:
            tmp += el.pnorm(2).power(2)

        tmp = tmp.sqrt().max()-1
        
        if tmp<=1e-5:
            res += 0
        else:
            res += np.inf

        return res
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """

        if not isinstance(x, BlockDataContainer):
            raise ValueError('proximal expected BlockDataContainer, got {}'.format(type(x)))

        res = x.containers[0].fill(0)

        for el in x.containers:
            tmp = el.pnorm(2).power(2)
            res += tmp
        res = res.sqrt()

        if has_numba and isinstance(tau, Number):
            try: 
                # may involve a copy if the data is not contiguous
                tmparr = np.asarray(res.as_array(), order='C', dtype=tmp.dtype)
                if _proximal_step_numba(tmparr, np.abs(tau)) != 0:
                    # if numba silently crashes
                    raise RuntimeError('MixedL21Norm.proximal: numba silently crashed.')
                
                res.fill(tmparr)
            except:
                res = _proximal_step_numpy(tmp, tau)
        else:
            res = _proximal_step_numpy(tmp, tau)

        if out is None:
            return res.multiply(x)

        else:
            res.multiply(x, out=out)


class MixedL21Norm(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 
        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        res = x[0].containers[0].get_uniform_copy(0)
        for el in x.containers:
            res += el.pnorm(2).power(2)
        return res.sqrt().sum()                              
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """

        # def wrong

        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        res = 0

        tmp = x.containers[0].copy().fill(0)

        for el in x.containers:
            tmp += el.pnorm(2).power(2)

        tmp = tmp.sqrt().max()-1
        
        if tmp<=1e-5:
            res += 0
        else:
            res += np.inf

        return res
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """

        if not isinstance(x, BlockDataContainer):
            raise ValueError('proximal expected BlockDataContainer, got {}'.format(type(x)))

        res = x.containers[0].fill(0)

        for el in x.containers:
            tmp = el.pnorm(2).power(2)
            res += tmp
        res = res.sqrt()

        if has_numba and isinstance(tau, Number):
            try: 
                # may involve a copy if the data is not contiguous
                tmparr = np.asarray(res.as_array(), order='C', dtype=tmp.dtype)
                if _proximal_step_numba(tmparr, np.abs(tau)) != 0:
                    # if numba silently crashes
                    raise RuntimeError('MixedL21Norm.proximal: numba silently crashed.')
                
                res.fill(tmparr)
            except:
                res = _proximal_step_numpy(tmp, tau)
        else:
            res = _proximal_step_numpy(tmp, tau)

        if out is None:
            return res.multiply(x)

        else:
            res.multiply(x, out=out)



class GuidedJTV(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, anat, **kwargs):

        super(GuidedJTV, self).__init__()  

        self.anat = anat
                    
    
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 
        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        
        res = x.pnorm(2).power(2)
        for bdc in self.anat:
            res += bdc.pnorm(2).power(2)

        return res.sqrt().sum()                              
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """

        # def wrong

        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 

        tmp = x.pnorm(2).power(2)

        for bdc in self.anat:
            tmp += bdc.pnorm(2).power(2)

        tmp = tmp.sqrt().max()-1
        
        if tmp<=1e-5:
            res += 0
        else:
            res += np.inf

        return res
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """

        if not isinstance(x, BlockDataContainer):
            raise ValueError('proximal expected BlockDataContainer, got {}'.format(type(x)))

        res = x.clone()
        tmp = res.pnorm(2).power(2)
        for bdc in self.anat:
            tmp += bdc.pnorm(2).power(2)
        tmp = tmp.sqrt()

        if has_numba and isinstance(tau, Number):
            try: 
                # may involve a copy if the data is not contiguous
                tmparr = np.asarray(res.as_array(), order='C', dtype=tmp.dtype)
                if _proximal_step_numba(tmparr, np.abs(tau)) != 0:
                    # if numba silently crashes
                    raise RuntimeError('MixedL21Norm.proximal: numba silently crashed.')
                
                res = tmp
                res.fill(tmparr)
            except:
                res = _proximal_step_numpy(tmp, tau)
        else:
            res = _proximal_step_numpy(tmp, tau.as_array())

        if out is None:
            res = x.multiply(res)
        else:
            x.multiply(res, out = out)
            res = out

        if out is None:
            return res
        
class SmoothGuidedJTV(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, anat, epsilon=0.001, **kwargs):

        super(SmoothGuidedJTV, self).__init__()  

        if isinstance(anat, list):
            self.anat = anat
        else:
            self.anat = [anat]
        self.anat_norm = [bdc.pnorm(2).power(2) for bdc in self.anat]
        self.epsilon = epsilon
                    
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 
        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
        

        res = x.pnorm(2).power(2) + self.epsilon**2

        for bdc in self.anat_norm:
            res += bdc

        res = res.sqrt().sum()

        return res     

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
                
        """           

        res = x.pnorm(2).power(2) + self.epsilon**2

        for bdc in self.anat_norm:
            res += bdc

        denom = res.sqrt()
        
        if out is None:
            return x.divide(denom)
        else:
            x.divide(denom, out=out)                  
            
                        

class SmoothMixedL21Norm(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''

        super(SmoothMixedL21Norm, self).__init__(L=1)
        self.epsilon = epsilon   
                
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 

        res = 0

        for el in x.containers:
            res += (el.pnorm(2).power(2) + self.epsilon**2).sqrt().sum()

        return res
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
              if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                   
        res = x.containers[0].copy().fill(0)

        for el in x.containers:
            res += el.pnorm(2).power(2)

        denom = (res + self.epsilon**2).sqrt()
        x.divide(denom, out = res)

        if out is None:
            return x.divide(denom)
        
        else:
            x.divide(denom, out = out)        
                
        """     
        
  

class SmoothGuidedTV_im_domain(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon, operator, anat_grad):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''

        super(SmoothGuidedTV_im_domain, self).__init__(L=1)
        self.epsilon = epsilon   
        self.anat_grad = anat_grad
        self.operator = operator
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.
        """

        tmp = self.operator.direct(x)
        res = x[0].clone()
        res.fill(0)

        for el in tmp.containers:
            res += el.pnorm(2).power(2)
        self.anat_grad.pnorm(2).power(2)
        res+= self.epsilon**2

        return res.sqrt().sum()
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
                
        """     
                   
        res = x[0].copy()
        res.fill(0)
        tmp = self.operator.direct(x)

        for el in tmp.containers:
            res += el.pnorm(2).power(2)
        res += self.anat_grad.pnorm(2).power(2)

        denom = (res + self.epsilon**2).sqrt()
        tmp.divide(denom, out = tmp)
        
        if out is None:
            return self.operator.adjoint(tmp)
        
        else:
            self.operator.adjoint(tmp, out = out)

class SmoothGuidedTV_im_domain_one(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon, operator, anat_grad):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''
        super(SmoothGuidedTV_im_domain_one, self).__init__(L=1)
        self.epsilon = epsilon   
        self.anat_grad = anat_grad
        self.operator = operator
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.
        """
        tmp = self.operator.direct(x)
        res = x.clone()
        res.fill(0)

        res += tmp.pnorm(2).power(2)
        for el in self.anat_grad:
            res += el.pnorm(2).power(2)
        res+= self.epsilon**2

        return res.sqrt().sum()
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
        """       
        res = x.copy()
        res.fill(0)
        tmp = self.operator.direct(x)

        res += tmp.pnorm(2).power(2)
        for el in self.anat_grad:
            res += el.pnorm(2).power(2)

        denom = (res + self.epsilon**2).sqrt()
        tmp.divide(denom, out = tmp)
        
        if out is None:
            return self.operator.adjoint(tmp)
        
        else:
            self.operator.adjoint(tmp, out = out)