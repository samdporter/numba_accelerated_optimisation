# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ZeroFunction, ConstantFunction
from cil.framework import BlockDataContainer
import numpy as np
import logging
from numbers import Number

import numpy.random

from sirf.STIR import TruncateToCylinderProcessor

import matplotlib.pyplot as plt

from numba import jit, njit, prange

@njit(parallel=True)
def divide_numba(a,b, eps=1e-4):
    res = np.zeros_like(a)
    tmp = res.ravel()
    for i in prange(a.size):
        if b.flat[i] == 0: 
            tmp[i] = a.flat[i]/eps
        else:
            tmp[i] = a.flat[i]/b.flat[i]
    return res

def divide(a,b):
    res = a.clone()
    res.fill(divide_numba(a.as_array(), b.as_array()))
    return res  

class Primal_Dual_3Splitting(Algorithm):
    

    r"""Base class for Primal Dual algorithms with 3 operators. Limited to Condat-Vu, PD3O and PDFB.
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A proximable convex function.
        h : Function
            A composite convex function.            

     """    
    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims = False, **kwargs):

        super(Primal_Dual_3Splitting, self).__init__(**kwargs)


        if isinstance(f, ZeroFunction) or isinstance(f, ConstantFunction) or f is None:
            logging.warning(" If self.f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwhise, select a relatively small parameter sigma")                           
        self.set_up(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims = save_ims, **kwargs)
 
                  
    def set_up(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims = False, **kwargs):
        logging.info("{} setting up".format(self.__class__.__name__, ))

        self.f = f # smooth function       
        self.g = g # proximable
        self.h = h # composite
        self.operator = operator
        
        self.tau = rho*tau
        self.sigma = rho*sigma 
        
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.initial = self.x.copy()

        self.x_bar = self.x.copy()    
        self.x_old = self.x.copy()
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)
                
        self.grad_f = self.operator.domain_geometry().allocate(0)

        if self.check_convergence() is False:
            logging.warning("The algorithm is not guaranteed to converge")

        self.save_ims = save_ims
  
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
        
    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """
        
        # following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0
        # #TODO if we change the order of proximal steps we recover the PDDY algorithm (dual) PDHG, when f=0
        
        # 

        self.grad_f =  self.calculate_gradient() # can be extended to stochastic & svrg

        self.s = self.h.proximal_conjugate(self.s_old + self.tau * self.operator.direct(self.x_bar), self.tau,)

        self.x = self.g.proximal(self.x_old - self.sigma*(self.grad_f-self.operator.adjoint(self.s)), self.sigma)

        self.update_xbar()

        self.update_previous_solution()

    def calculate_gradient(self):
        return self.f.gradient(self.x)
    
    def recalculate_sigma(self):
        raise NotImplementedError
    
    def recalculate_tau(self):
        raise NotImplementedError

    def update_xbar(self):
        raise NotImplementedError

    def update_previous_solution(self):
        self.x_old.fill(self.x)      
        self.s_old.fill(self.s) 
                                                                        
    def update_objective(self):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g + fun_h

        try:

            d1 = - self.h.convex_conjugate(self.s)
            tmp = self.operator.adjoint(self.s)
            tmp *= -1
            d2 = -self.g.convex_conjugate(tmp)
            d3 = -self.f.convex_conjugate(tmp)

            d = d1+d2+d3
    
            self.loss.append([p1, d, p1-d, fun_f, fun_g, fun_h])

        except:

            self.loss.append([p1, np.nan, np.nan, fun_f, fun_g, fun_h])
        


        if self.save_ims:
            self.x.write(f"x_{self.iteration}")

    def plot_image(self):
        plt.figure()
        plt.imshow(self.x.as_array()[32])
        plt.colorbar()
        plt.show()
    
    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]
    @property
    def f_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def r_objective(self):
        return [x[2] for x in self.loss]
    
    @property
    def h_objective(self):
        return [x[3] for x in self.loss]
    

class Condat_Vu(Primal_Dual_3Splitting):
    """ Primal Dual Condat-Vu algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims=False, **kwargs):
    
        super(Condat_Vu, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims=save_ims, **kwargs)
        
    def update_xbar(self):
        self.x_bar = 2*self.x - self.x_old

    def check_convergence(self):
        if self.sigma*self.tau*self.operator.norm()**2 + self.sigma/(2*self.f.L) > 1:
            return False
        else:
            return True

class PDFP(Primal_Dual_3Splitting):
    """ Primal Dual Forward-Backward-Predictor algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims=False, **kwargs):
        
        super(PDFP, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims=save_ims, **kwargs)
        
    def update_xbar(self):
        self.x_bar = self.g.proximal(self.x_old - self.sigma*(self.grad_f-self.operator.adjoint(self.s)), self.sigma)

    def check_convergence(self):
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            return False
        elif self.sigma >= 2/self.f.L:
            return False
        else:
            return True

class PD3O(Primal_Dual_3Splitting):
    """ Primal Dual 3 Operator algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=0.999, initial=None, save_ims=False, **kwargs):
        
        super(PD3O, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims = save_ims, **kwargs)

        self.rho = rho
        self.grad_f_old = self.initial.clone()
        
    def update_xbar(self):
        self.x_bar = 2*self.x - self.x_old + self.sigma*(self.grad_f_old-self.grad_f)

    def update_previous_solution(self):
        
        self.x_old.fill(self.x)      
        self.s_old.fill(self.s) 
        self.grad_f_old.fill(self.grad_f)

    def check_convergence(self):
        if not isinstance(self.sigma, Number) or not isinstance(self.tau, Number):
            print("Unable to check convergence, sigma and tau must be numbers for this")
            return
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            print(f"sigma*tau*norm(A)^2 = {self.sigma*self.tau*self.operator.norm()**2} >= 1")
            return False
        elif self.sigma >= 2/self.f.L:
            print(f"sigma = {self.sigma} >= 2/L = {2/self.f.L}")
            return False
        else:
            return True

########################### 
### Stochastic Variants ###
###########################    
# 
# Currently no variance reduction is implemented - probably needed at some point    

class SPD3O(PD3O):
    """ Stochastic Primal Dual 3 Operator algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=0.999, initial=None, save_ims=False, **kwargs):
        
        super(SPD3O, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims=save_ims, **kwargs)

    def calculate_gradient(self):
        i = np.random.randint(0, len(self.f))
        return self.f[i].gradient(self.x)
    
    def update_objective(self):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = sum([f(self.x) for f in self.f])
        p1 = fun_f + fun_g + fun_h

        try:
            d1 = - self.h.convex_conjugate(self.s)
            tmp = self.operator.adjoint(self.s)
            tmp *= -1
            d1 -= self.g.convex_conjugate(tmp)
            d1 += self.f.convex_conjugate(tmp)
        
            self.loss.append([p1, fun_f, fun_g, fun_h])
        
        except:
            self.loss.append([p1, fun_f, fun_g, fun_h])

        if self.save_ims:
            self.x.write(f"x_{self.iteration}")

    def check_convergence(self):

        return True
        # find maximum L from self.f
        L = max([f.L for f in self.f])
        if not isinstance(self.sigma, Number) or not isinstance(self.tau, Number):
            print("Unable to check convergence, sigma and tau must be numbers for this")
            return
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            print(f"sigma*tau*norm(A)^2 = {self.sigma*self.tau*self.operator.norm()**2} >= 1")
            return False
        elif self.sigma >= 2/L:
            print(f"sigma = {self.sigma} >= 2/L = {2/L}")
            return False
        else:
            return True

    
class SPDFP(PDFP):
    """ Stochastic Primal Dual Forward-Backward-Predictor algorithm """
    
    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims=False, **kwargs):
        
        super(SPDFP, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims=save_ims, **kwargs)

    def calculate_gradient(self):
        i = np.random.randint(0, len(self.f))
        return self.f[i].gradient(self.x)

    def update_objective(self):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = sum([f(self.x) for f in self.f])
        p1 = fun_f + fun_g + fun_h
        
        self.loss.append([p1, fun_f, fun_g, fun_h])

        if self.save_ims:
            self.x.write(f"x_{self.iteration}")

    def check_convergence(self):
        # find maximum L from self.f
        L = max([f.L for f in self.f])
        if not isinstance(self.sigma, Number) or not isinstance(self.tau, Number):
            print("Unable to check convergence, sigma and tau must be numbers for this")
            return
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            print(f"sigma*tau*norm(A)^2 = {self.sigma*self.tau*self.operator.norm()**2} >= 1")
            return False
        elif self.sigma >= 2/L:
            print(f"sigma = {self.sigma} >= 2/L = {2/L}")
            return False
        else:
            return True
    
class SCondat_Vu(Condat_Vu):
    """ Stochastic Primal Dual Condat-Vu algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=None, initial=None, save_ims=False, **kwargs):
    
        super(SCondat_Vu, self).__init__(f=f, g=g, h=h, operator=operator, tau=tau, sigma=sigma, rho=rho, initial=initial, save_ims=save_ims, **kwargs)

    def calculate_gradient(self):
        i = np.random.randint(0, len(self.f))
        return self.f[i].gradient(self.x)
    
    def update_objective(self):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = sum([f(self.x) for f in self.f])
        p1 = fun_f + fun_g + fun_h
        
        self.loss.append([p1, fun_f, fun_g, fun_h])

        if self.save_ims:
            self.x.write(f"x_{self.iteration}")

    def check_convergence(self):
        # find maximum L from self.f
        L = max([f.L for f in self.f])
        if self.sigma*self.tau*self.operator.norm()**2 + self.sigma/(L) > 1:
            return False
        else:
            return True


        


        



        


#######################
### Here be dragons ###
#######################

class new_SPD3O(Algorithm):
    """ Primal Dual 3 Operator algorithm """

    def __init__(self, f, g, h, operator, tau=None, sigma=None, rho=0.999, gamma=None, initial=None, probs = None, am=None, svrg=True, **kwargs):

        ## Can improve this later

        super(SPD3O, self).__init__(**kwargs)

        self.f = f
        self.g = g
        self.h = h

        self.g = g

        self.am = am

        self.current_subset = 0

        self.operator = operator

        self.svrg = svrg

        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.initial = self.x.copy()

        self.x_bar = self.x.copy()    
        self.x_old = self.x.copy()
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)

        self.sens = kwargs.get('sens', None)
        self.norm = operator.norm()

        self.cyl = TruncateToCylinderProcessor()
        self.cyl.set_strictly_less_than_radius(True)

        if probs == None:
            self.probs = [1/len(sigma) for _ in range(len(sigma))]
        else:
            self.probs = probs

        self.rho = rho
        self.gamma = gamma
        self.tau = [rho*t for t in tau]
        self.sigma = [rho*s for s in sigma]

        self.grad_f = [self.operator.domain_geometry().allocate(0) for _ in range(len(self.sigma))]
        self.grad_f_old = [self.operator.domain_geometry().allocate(0) for _ in range(len(self.sigma))]
        if self.svrg:
            self.grad_w = [self.operator.domain_geometry().allocate(0) for _ in range(len(self.sigma))]
            self.grad_full = self.operator.domain_geometry().allocate(0)

        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))     

    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """
        
        # following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0
        # #TODO if we change the order of proximal steps we recover the PDDY algorithm (dual) PDHG, when f=0
        
        # 

        i = numpy.random.choice(len(self.sigma), p=self.probs)
        self.current_subset = i

        self.h.proximal_conjugate(self.s_old + self.tau[i] * self.operator.direct(self.x_bar), self.tau[i], out = self.s)

        self.g.proximal(self.x_old - self.sigma[i]*(self.grad_f[i]-self.operator.adjoint(self.s)), self.sigma[i], out = self.x)

        self.grad_f[i] =  self.calculate_gradient(i) # can be extended to stochastic & svrg

        self.update_xbar(i)

        print(f"\n update {self.iteration} complete, subset {i} \n")


    def calculate_gradient(self, i):
        if self.svrg:
            g = self.am[i].adjoint(self.f[i].gradient(self.am[i].direct(self.x_bar)))
            return g - self.grad_w[i] + self.grad_full
        return self.am[i].adjoint(self.f[i].gradient(self.am[i].direct(self.x)))

    def update_xbar(self, i):
        self.x_bar = 2*self.x - self.x_old + self.sigma[i]*(self.grad_f_old[i]-self.grad_f[i])

    def update_previous_solution(self):
        
        self.x_old.fill(self.x)      
        self.s_old.fill(self.s) 
        self.grad_f_old[self.current_subset].fill(self.grad_f[self.current_subset])

    def update_gradient(self):
    
        print("\n calculating full gradient \n")

        self.grad_full.fill(0)
        for i, f in enumerate(self.f):
            self.grad_f_old[i] = self.am[i].adjoint(f.gradient(self.am[i].direct(self.x)))
            self.grad_full += self.grad_f_old[i]
        self.grad_full/=len(self.f)
    
        print("\n finished with gradient calculation \n")


    def check_convergence(self):
        if not isinstance(self.sigma, Number) or not isinstance(self.tau, Number):
            print("Unable to check convergence, sigma and tau must be numbers for this")
            return
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            print(f"sigma*tau*norm(A)^2 = {self.sigma*self.tau*self.operator.norm()**2} >= 1")
            return False
        elif self.sigma >= 2/self.f.L:
            print(f"sigma = {self.sigma} >= 2/L = {2/self.f.L}")
            return False
        else:
            return True
        
    def recalculate_step_sizes(self):
        self.recalculate_sigma()
        self.recalculate_tau()
        for i in range(len(self.sigma)):
            self.sigma[i]+=1e-8
            self.tau[i]+=1e-8
            self.sigma[i]*=self.rho
            self.tau[i]*=self.rho
        #self.sigma[0].write(f"sigma_{self.iteration}_{i}")
        #self.tau[0].write(f"tau_{self.iteration}_{i}")
        
    def recalculate_sigma(self):
        self.sigma = [(self.gamma*(divide(self.x, s)).minimum(2/f.L)) for s, f in zip(self.sens, self.f)]
        for i, s in enumerate(self.sigma):
            self.cyl.apply(s)
            #print(f"max sigma {i}: {2/self.f[i].L}")

    def recalculate_tau(self):
        self.tau = [s/self.norm**2 for s in self.sigma]
        #print(f"tau max = {self.tau.max()}")

    def update_objective(self):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
                 
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = 0
        for i in range(len(self.f)):
            fun_f += self.f[i](self.am[i].direct(self.x))
        p1 = fun_f + fun_g + fun_h

        self.recalculate_step_sizes()
        
        self.loss.append([p1, fun_f, fun_g, fun_h])

        if self.svrg:
            self.update_gradient()

        self.save_images()

    def save_images(self):
        if isinstance(self.x, BlockDataContainer):
            for i, im in enumerate(self.x.containers):
                im.write(f"x_{i}_{self.iteration}")
        else:
            self.x.write(f"x_{self.iteration}")
        
        
class PDHG(Primal_Dual_3Splitting):
    """ Primal Dual Hybrid Gradient"""

    def __init__(self, g, h, operator, tau=None, sigma=None, rho=0.999, initial=None, **kwargs):
        super().__init__(g, h, operator, tau, sigma, rho, initial, **kwargs)    

        self.f = ConstantFunction(0)

    def update_xbar(self):
        self.x_bar = 2*self.x - self.x_old

    def check_convergence(self):
        if self.sigma*self.tau*self.operator.norm()**2 >= 1:
            return False
        else:
            return True
















        
class SCondat_Vu(Algorithm):
    

    r"""Primal Dual Three Operator Splitting (PD3O) algorithm, see "A New Primal–Dual Algorithm for Minimizing the Sum
        of Three Functions with a Linear Operator"
        https://link.springer.com/article/10.1007/s10957-022-02061-8
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A proximable convex function.
        h : Function
            A composite convex function.            

     """    


    def __init__(self, f, r, h, operator, prob, tau=None, sigma=None, gamma=None, initial=None, 
                 sensitivities=None, svrg=True, precond=False, **kwargs):

        super(SCondat_Vu, self).__init__(**kwargs)

        self.f = f # smooth function
        if isinstance(self.f, ZeroFunction):
            logging.warning(" If self.f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwhise, select a relatively small parameter sigma")                        
        
        self.r = r # proximable
        self.h = h # composite
        self.operator = operator
        
        self.tau = tau
        self.sigma = sigma    
        self.prob = prob     

        self.set_up(f=f, r=r, h=h, operator=operator, tau=tau, sigma=sigma, gamma=gamma, 
                    initial=initial, sensitivities=sensitivities,svrg = svrg, precond=precond,
                    **kwargs)
 
                  
    def set_up(self, tau=None, sigma=None, gamma=None, sensitivities=None, 
               initial=None, svrg=True,precond=False, **kwargs):

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.x_bar = self.x.copy()    
        self.x_old = self.operator.domain_geometry().allocate(0)
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)

        self.sensitivities = sensitivities
        self.precond = precond
        
        self.svrg = svrg

        if self.svrg:
            self.grad = self.operator.domain_geometry().allocate(0)
            self.g_old = [self.operator.domain_geometry().allocate(0) for _ in self.prob]

        self.norm = self.operator.norm()
                
        if gamma is None:
            self.gamma = 1

        if self.sensitivities is not None:
            if sigma is None:
                self.sigma = [self.operator.domain_geometry().allocate(0) for _ in self.prob]
            if tau is None:
                self.tau = self.operator.range_geometry().allocate(0)
                self.ones = self.operator.domain_geometry().allocate(1)
        else:
            if self.sigma is None:
                self.tau = self.sigma/self.norm**2

        if isinstance(self.sigma, Number):
            self.sigma = [self.sigma for _ in self.prob]
            if self.tau is None:
                self.tau = [self.sigma/self.norm**2 for _ in self.prob]
        if isinstance(self.tau, Number):
            self.tau = [self.tau for _ in self.prob]
  
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))     

    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """
        
        # following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0
        # #TODO if we change the order of proximal steps we recover the PDDY algorithm (dual) PDHG, when f=0
    
        # choose subset gradient and update gradient
        i = int(np.random.choice(len(self.prob), 1, p=self.prob))
        if self.svrg and self.iteration>len(self.prob):
            g_new = -self.f[i].gradient(self.x)
            g = g_new - self.g_old[i] + self.grad
        else:
            g = -self.f[i].gradient(self.x) #(negative as sirf of
            
        # dual proximal step
        self.s = self.h.proximal_conjugate(self.s + self.tau[i]*self.operator.direct(self.x_bar), self.tau[i])

        # gradient step
        self.x -= self.sigma[i]*g 
        
        # primal proximal step
        self.x = self.r.proximal(self.x - self.sigma*self.operator.adjoint(self.s), self.sigma)

        # over relaxation
        self.x_bar = 2*self.x - self.x_old

        # update old variables
        self.x_old.fill(self.x)
                                                      
    def update_objective(self):
        """
        Evaluates the primal objective
        """        
                 
        fun_h = self.h(self.operator.direct(self.x))
        fun_r = self.r(self.x)
        fun_f = 0
        for f in self.f:
            fun_f -= f(self.x)
        p1 = fun_f + fun_r + fun_h

        self.save_images()

        if self.sensitivities is not None:
            if self.precond:
                self.update_step_sizes_diag()
            else:
                self.update_step_sizes()

        if self.svrg:
            self.update_gradient()

        self.loss.append([p1, fun_f, fun_r, fun_h])
        
    def save_images(self):
        if isinstance(self.x, BlockDataContainer):
            for i, x in enumerate(self.x.containers):
                x.write(f"output/x_{self.iteration}_{i}.hv")
                if self.svrg:
                    self.grad[i].write(f"output/g_{self.iteration}_{i}.hv")
        else:
            self.x.write(f"output/x_{self.iteration}.hv")
            if self.svrg:
                self.grad.write(f"output/g_{self.iteration}.hv")

    def update_step_sizes_diag(self):
        print("recalculating step sizes")
        self.tau = [0]*len(self.sensitivities)
        for i, s in enumerate(self.sensitivities):
            self.sigma[i] = divide(self.x, s)*self.prob[i]
            self.tau[i] = self.prob[i]*divide(self.ones, self.sigma[i])/self.norm**2

    def update_step_sizes(self):
        print("recalculating step sizes")
        self.tau = [0]*len(self.prob)
        for i, s in enumerate(self.sensitivities):
            bsrem = divide(self.x, s).norm()
            self.sigma[i] = self.gamma[i]*bsrem/self.norm
            self.tau[i] = 1/(self.norm*self.gamma[i]*bsrem)

    def update_gradient(self):

        print("calculating full gradient")

        self.grad.fill(0)
        for i, f in enumerate(self.f):
            self.g_old[i] = -f.gradient(self.x)
            self.grad += self.g_old[i]
        self.grad/=len(self.f.functions)

    
    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]
    @property
    def f_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def r_objective(self):
        return [x[2] for x in self.loss]
    
    @property
    def h_objective(self):
        return [x[3] for x in self.loss]
    
class gSPD3O(Algorithm):
    

    r"""Primal Dual Three Operator Splitting (PD3O) algorithm, see "A New Primal–Dual Algorithm for Minimizing the Sum
        of Three Functions with a Linear Operator"
        https://link.springer.com/article/10.1007/s10957-022-02061-8
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A proximable convex function.
        h : Function
            A composite convex function.            

     """    


    def __init__(self, f, r, h, operator, prob, tau=None, sigma=None, gamma=None, initial=None, 
                 sensitivities=None, svrg=True, precond=False, **kwargs):

        super(gSPD3O, self).__init__(**kwargs)

        self.f = f # smooth function
        if isinstance(self.f, ZeroFunction):
            logging.warning(" If self.f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwhise, select a relatively small parameter sigma")                        
        
        self.r = r # proximable
        self.h = h # composite
        self.operator = operator
        
        self.tau = tau
        self.sigma = sigma    
        self.prob = prob     

        self.set_up(f=f, r=r, h=h, operator=operator, tau=tau, sigma=sigma, gamma=gamma, 
                    initial=initial, sensitivities=sensitivities,svrg = svrg, precond=precond,
                    **kwargs)
 
                  
    def set_up(self, f, r, h, operator, tau=None, sigma=None, gamma=None, sensitivities=None, 
               initial=None, svrg=True,precond=False, **kwargs):

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.x_bar = self.x.copy()    
        self.x_old = self.operator.domain_geometry().allocate(0)
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)

        self.sensitivities = sensitivities
        self.precond = precond
        
        self.svrg = svrg

        if self.svrg:
            self.grad = self.operator.domain_geometry().allocate(0)
            self.g_old = [self.operator.domain_geometry().allocate(0) for _ in self.prob]

        self.norm = self.operator.norm()
                
        if gamma is None:
            self.gamma = 1

        if self.sensitivities is not None:
            if sigma is None:
                self.sigma = [self.operator.domain_geometry().allocate(0) for _ in self.prob]
            if tau is None:
                self.tau = self.operator.range_geometry().allocate(0)
                self.ones = self.operator.domain_geometry().allocate(1)
        else:
            if self.sigma is None:
                self.tau = self.sigma/self.norm**2

        if isinstance(self.sigma, Number):
            self.sigma = [self.sigma for _ in self.prob]
            if self.tau is None:
                self.tau = [self.sigma/self.norm**2 for _ in self.prob]
        if isinstance(self.tau, Number):
            self.tau = [self.tau for _ in self.prob]
  
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))     

    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """
        
        # following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0
        # #TODO if we change the order of proximal steps we recover the PDDY algorithm (dual) PDHG, when f=0
    
        # subset gradient step
        i = int(np.random.choice(len(self.prob), 1, p=self.prob))

        if self.svrg and self.iteration>len(self.prob):
            g_new = -self.f[i].gradient(self.x)
            g = g_new - self.g_old[i] + self.grad
        else:
            g = -self.f[i].gradient(self.x) #(negative as sirf of)

        # over relaxation step
        w = 2*self.x - self.x_bar - self.sigma[i]*g # sigma can be an image

        # proximal conjugate step
        self.s = self.tau[i]*self.operator.direct(w - self.sigma[i]*self.operator.adjoint(self.s)) # sigma can be an image, tau can be a dual variable
        self.s = self.h.proximal_conjugate(self.s, self.tau[i]) # tau can be a dual variable

        # update step
        self.x_bar = self.x - self.sigma[i]*(g - self.operator.adjoint(self.s))   # sigma[i] can be an image

        # proximal step
        self.x = self.r.proximal(self.x_bar, self.sigma[i])  # sigma can be an image

        print(f"iteration {self.iteration} - subset {i}")
                                                                        
    def update_objective(self):
        """
        Evaluates the primal objective
        """        
                 
        fun_h = self.h(self.operator.direct(self.x))
        fun_r = self.r(self.x)
        fun_f = 0
        for f in self.f:
            fun_f -= f(self.x)
        p1 = fun_f + fun_r + fun_h

        self.save_images()

        if self.sensitivities is not None:
            if self.precond:
                self.update_step_sizes_diag()
            else:
                self.update_step_sizes()

        if self.svrg:
            self.update_gradient()

        self.loss.append([p1, fun_f, fun_r, fun_h])
        
    def save_images(self):
        if isinstance(self.x, BlockDataContainer):
            for i, x in enumerate(self.x.containers):
                x.write(f"output/x_{self.iteration}_{i}.hv")
                if self.svrg:
                    self.grad[i].write(f"output/g_{self.iteration}_{i}.hv")
        else:
            self.x.write(f"output/x_{self.iteration}.hv")
            if self.svrg:
                self.grad.write(f"output/g_{self.iteration}.hv")

    def update_step_sizes_diag(self):
        print("recalculating step sizes")
        self.tau = [0]*len(self.sensitivities)
        for i, s in enumerate(self.sensitivities):
            self.sigma[i] = divide(self.x, s)*self.prob[i]
            self.tau[i] = self.prob[i]*divide(self.ones, self.sigma[i])/self.norm**2

    def update_step_sizes(self):
        print("recalculating step sizes")
        self.tau = [0]*len(self.prob)
        for i, s in enumerate(self.sensitivities): 
            bsrem = divide(self.x, s).norm()
            self.sigma[i] = self.gamma[i]*bsrem/self.norm
            self.tau[i] = 1/(self.norm*self.gamma[i]*bsrem)

    def update_gradient(self):

        print("calculating full gradient")

        self.grad.fill(0)
        for i, f in enumerate(self.f):
            self.g_old[i] = -f.gradient(self.x)
            self.grad += self.g_old[i]
        self.grad/=len(self.f.functions)

    
    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]
    @property
    def f_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def r_objective(self):
        return [x[2] for x in self.loss]
    
    @property
    def h_objective(self):
        return [x[3] for x in self.loss]
    
from numba import njit, prange

@njit(parallel=True)
def divide_numba(a,b):
    res = np.zeros_like(a)
    tmp = res.ravel()
    for i in prange(a.size):
        if b.flat[i] == 0:
            tmp[i] = 0.001
        else:
            tmp[i] = a.flat[i]/b.flat[i]
    return res

def divide(a,b):
    res = a.clone()
    res.fill(divide_numba(a.as_array(), b.as_array()))
    return res       



