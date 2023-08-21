import numpy as np
from numba import njit, prange
from python_optimisation.misc import project_inner
from python_optimisation.functions import ZeroFunction, SumFunction

######################
## Algorithm Class ###
######################

class Algorithm():

    def __init__(self, save=False) -> None:
        self.loss = []
        self.iteration = 0
        self.save = save
        self.running = True

    def set_up(self):
        raise NotImplementedError
    
    def run(self, verbose=1):
        self.update_objective()
        if verbose != 0:
            print(f'Iteration {0}/{self.max_iterations}')
            if verbose == 1:
                print(f'Loss: {self.loss[-1][0]}')
            elif verbose == 2:
                print(f'Loss: {self.loss[-1]}')
        for i in range(self.max_iterations):
            self.iteration = i
            self.update()
            if self.iteration % self.update_objective_interval == 0:
                self.update_objective()
                if verbose != 0:
                    print(f'Iteration {i+1}/{self.max_iterations}')
                    if verbose == 1:
                        print(f'Loss: {self.loss[-1][0]}')
                    elif verbose == 2:
                        print(f'Loss: {self.loss[-1]}')

    def update(self):
        raise NotImplementedError

    def stopping_criteria(self):
        self.running = False
    
    def update_objective(self, verbose):
        raise NotImplementedError

    @property
    def solution(self):
        return self.x

    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]

    @property
    def time(self):
        raise NotImplementedError   
    
######################################
###### Algorithm Classes #############
######################################

######################
## 3 operator PD #####
######################

class Primal_Dual_3Splitting(Algorithm):
    
    """ 
    sigma: primal step size <= 2/f.L
    tau: dual step size
    sigma * tau * ||L||^2 <1
    """

    def set_up(self, initial, f, g, h, operator, sigma, tau, rho=1, 
               max_iterations=10, update_interval=1, save = False):
        
        self.f = f
        self.g = g
        self.h = h
        self.operator = operator

        assert rho*sigma <= 2/f.L, "Condition sigma <= 2/f.L not met"
        assert sigma * tau * operator.norm()**2 < 1, "Condition sigma * tau * ||L||^2 < 1 not met"

        self.tau = rho*tau
        self.sigma = rho*sigma

        self.max_iterations = max_iterations
        self.save_interval = update_interval
        self.save = save

        self.x = initial
        self.x_bar = self.x.copy()
        self.x_old = self.x.copy()

        self.s = self.operator.direct(self.x)
        self.s_old = self.s.copy()

        self.grad = initial.copy()
    
    def run(self, verbose=0):
        if self.save:
            self.save()
        self.update_objective(verbose)
        while self.running:
            for _ in range(self.max_iterations):
                self.iteration+=1
                self.update()
                if np.min(self.x) < 0 and np.min(self.x) > -1e-10:
                    self.x = np.maximum(self.x,0)
                if self.iteration % self.save_interval == 0:
                    self.update_objective(verbose)
            self.running = False

    def update(self):

        self.grad_f =  self.calculate_gradient() # can be extended to stochastic & svrg

        self.s = self.h.proximal_conjugate(self.s_old + self.tau * self.operator.direct(self.x_bar), self.tau,)

        self.x = self.g.proximal(self.x_old - self.sigma*(self.grad_f-self.operator.adjoint(self.s)), self.sigma,)   

        self.update_xbar()

        self.update_previous_solution()

    def calculate_gradient(self):
        return self.gradient()

    def gradient(self):
        return self.f.gradient(self.x)
    
    def update_previous_solution(self):
        self.x_old = self.x.copy()      
        self.s_old = self.s.copy()

    def update_objective(self, verbose):
        """
        Evaluates the primal objective
        """        

        #self.plot_image()
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g + fun_h
        
        self.loss.append([p1, fun_f, fun_g, fun_h])
        if verbose != 0:
            print(f"iteration: {self.iteration}")
            print(f"Objectives: {self.loss[-1]}")

    def save(self):
        np.save(f"x_{self.iteration}", self.x)
        
    @property
    def f_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def r_objective(self):
        return [x[2] for x in self.loss]
    
    @property
    def h_objective(self):
        return [x[3] for x in self.loss]
    
class PD3O(Primal_Dual_3Splitting):
    """ Primal Dual 3 Operator algorithm """

    def __init__(self):
        
        super(PD3O, self).__init__()

    def set_up(self, initial, f, g, h, operator, tau, sigma, rho=1, max_iterations=10, save_interval=1):
        
        self.f = f
        self.g = g
        self.h = h
        self.operator = operator

        self.tau = rho*tau
        self.sigma = rho*sigma

        self.max_iterations = max_iterations
        self.save_interval = save_interval

        self.x = initial
        self.x_bar = self.x.copy()
        self.x_old = self.x.copy()

        self.s = self.operator.direct(self.x)
        self.s_old = self.s.copy()

        self.grad = initial.copy()
        self.grad_f_old = initial.copy()
        
    def update_xbar(self):
        self.x_bar = 2*self.x - self.x_old + self.sigma*(self.grad_f_old-self.grad_f)

    def update_previous_solution(self):
        
        self.x_old = self.x.copy()
        self.s_old = self.s.copy()
        self.grad_f_old = self.grad_f.copy()

######################
## Simple PDHG #######
######################

class PDHGAlgorithm(Algorithm):

    def __init__(self, f, g, operator, primal_step_size, dual_step_size, rho=0.99,
                 max_iterations=100, save_interval=1, save=False):
        super().__init__(save=save)
        self.operator = operator
        self.primal_step_size = primal_step_size
        self.dual_step_size = dual_step_size
        self.f = f
        self.g = g
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.y = None  # Dual variable
        self.bar_x = None  # Over-relaxed primal variable

        self.rho = rho
        self.primal_value = 0

    def set_up(self, initial_x, initial_y):
        self.x = initial_x
        self.y = initial_y
        self.bar_x = initial_x.copy()
        self.x_old = initial_x.copy()
        self.optimal = self.x.copy()

    def update(self):

        self.y = self.f.proximal_conjugate(self.y + self.dual_step_size * 
                                           self.operator.direct(self.bar_x), self.dual_step_size,)
        self.x = self.g.proximal(self.x - self.primal_step_size * 
                                    self.operator.adjoint(self.y), self.primal_step_size,)
        self.x_old = self.x.copy()

        self.bar_x = self.x + self.rho * (self.x - self.x_old)

    def update_objective(self):
        primal_g = self.g(self.x)
        primal_f = self.f(self.operator.direct(self.x))

        if primal_f+primal_g < self.primal_value:
            self.optimal = self.x.copy()

        self.primal_value = primal_f + primal_g

        self.loss.append((self.primal_value,))

    def stopping_criteria(self):
        # Implement stopping criteria if needed
        pass

    @property
    def time(self):
        # Implement time-related functionality if needed
        pass


######################
## PDHG 3 Splitting ##
######################

class PDHG(Primal_Dual_3Splitting):
    """ Primal Dual Hybrid Gradient algorithm """

    def __init__(self):
        
        super(PDHG, self).__init__()

    def set_up(self, initial, g, h, operator, tau, sigma, theta = 1, rho=0.999,
               adaptivity = 0, max_iterations=10, save_interval=1, adapt_interval=1,
               lower_bound=0):

        self.f = ZeroFunction()
        self.g = g
        self.h = h
        self.operator = operator

        self.tau = rho*tau
        self.sigma = rho*sigma
        self.theta = theta
        self.adaptivity = adaptivity
        self.adapt_interval = adapt_interval
        self.lower_bound = lower_bound

        self.max_iterations = max_iterations
        self.save_interval = save_interval

        self.x = initial
        self.x_bar = self.x.copy()
        self.x_old = self.x.copy()

        self.s = self.operator.direct(self.x)
        self.s_old = self.s.copy()

        if self.adaptivity == 0:
            self.beta = 1.01
            self.eta = 0.05
            self.c = 0.8
            self.xi = 0.9
        elif self.adaptivity ==1:
            self.alpha = 0.95
            self.eta = 0.95
            self.c = 0.9

    def update(self):

        self.s = self.h.proximal_conjugate(self.s_old + self.tau * self.operator.direct(self.x_bar), self.tau,)
        self.x = self.g.proximal(self.x_old - self.sigma*self.operator.adjoint(self.s), self.sigma,)

        self.update_xbar()

        if self.lower_bound is not None:
            self.x = np.maximum(self.x, self.lower_bound)

        # Update the step sizes when iteration is a multiple of save_interval
        if self.iteration % self.adapt_interval == 0:
            if self.adaptivity == 0:
                # Adaptive step size update
                self.adaptive_step_sizes()
            elif self.adaptivity == 1:
                # Backtracking step size update
                self.backtracking()
            try:
                if self.objective[-1] > self.objective[-2]:
                    if self.adaptivity is not None:
                        self.adaptivity = None
                    else:
                        self.running = False
            except:
                pass

        self.update_previous_solution()

    def backtracking(self):        
        """ New adaptive step size update with Numba optimizations """

        # Extract necessary attributes
        gamma1, gamma2 = self.sigma, self.tau
        x_old, y_old = self.x_old, self.s_old
        x, y = self.x, self.s
        alpha = eta = 0.95

        # Compute B

        x_diff_dir = self.operator.direct(x-x_old)
        
        B = _compute_B(self.c, gamma1, x, x_old, gamma2, y, y_old, x_diff_dir)

        # Update conditions
        if B <= 0:
            gamma1 /= 2
            gamma2 /= 2
        
        if custom_norm(x)**2 >= 2 * custom_norm(y)**2:
            gamma1 /= (1 - self.alpha)
            gamma2 *= (1 - self.alpha)
            alpha *= eta
        elif custom_norm(y)**2 >= 2 * custom_norm(x)**2:
            gamma2 /= (1 - self.alpha)
            gamma1 *= (1 - self.alpha)
            self.alpha *= self.eta

        # Save updates to attributes
        self.sigma, self.tau = gamma1, gamma2


    def adaptive_step_sizes(self):        
        """ 
        Adaptive step size update
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8282164
        """

        self.calculate_residual() # Calculate the residuals for primal and dual variables

        if np.sum(self.p) != 0 and np.sum(self.d) != 0:

            # Calculate the weights
            self.w_p = _compute_w(self.x_old, self.x, self.p)
            self.w_d = _compute_w(self.s_old, self.s, self.d)

            # Calculate the ratio between the residuals
            R_pd = _compute_R_PD(self.p, self.d)

            # Update adaptive parameters
            if self.w_p > self.c:
                self.sigma *= self.beta
            if self.w_p < 0:
                self.sigma *= self.xi
            if self.w_d > self.c:
                self.tau *= self.beta
            if self.w_d < 0:
                self.tau *= self.xi

            # Update the step sizes
            self.sigma *= R_pd**self.eta
            self.tau *= R_pd**(-self.eta)

    def calculate_residual(self):
        """ Calculate the residuals for primal and dual variables """
        self.p = 1/self.sigma * ((self.x_old - self.x) - self.operator.adjoint(self.s_old-self.s))
        self.d = 1/self.tau * ((self.s_old - self.s) + self.theta*self.operator.direct(self.x_old-self.x))

        #"p: ", np.sum(self.p))
        #print("d: ", np.sum(self.d))
        
    def update_xbar(self):
        self.x_bar = self.x + self.theta*(self.x -self.x_old)

@njit
def custom_norm(arr):
    s = 0.0
    for val in np.nditer(arr):  # Iterates over all elements in the array, irrespective of its dimension
        s += val**2
    return np.sqrt(s)

@njit(parallel=True)
def _compute_w(k, k1, pd):
    # Ensure that arrays are flattened for dot product
    k_flat = k.ravel()
    k1_flat = k1.ravel()
    pd_flat = pd.ravel()

    # Compute dot product manually for parallelism
    dot_product = 0.0
    for i in prange(k_flat.size):
        dot_product += (k_flat[i] - k1_flat[i]) * pd_flat[i]  # Notice a bracket correction here

    norm_diff = custom_norm(k - k1)
    norm_pd = custom_norm(pd)
    
    if norm_diff * norm_pd == 0:
        return 0  # Handle division by zero
    
    w_k1_P = dot_product / (norm_diff * norm_pd)
    return w_k1_P

@njit
def _compute_R_PD(p_k1, d_k1):
    norm_p = custom_norm(p_k1)
    norm_d = custom_norm(d_k1)
    
    # Handle division by zero
    if norm_d == 0:
        return 0  # or return np.inf or any other placeholder value
    
    R_k1_PD = norm_p / norm_d
    return R_k1_PD

@njit
def _compute_B(c, gamma1, x, x_old, gamma2, y, y_old, x_diff_dir):

    term1 = c / (2 * gamma1) * custom_norm(x - x_old)**2
    term2 = c / (2 * gamma2) * custom_norm(y - y_old)**2
    term3 = 2 * np.dot((y - y_old).ravel(), x_diff_dir.ravel())

    return term1 + term2 - term3


class FISTA(Algorithm):
    """ Fast Iterative Shrinkage Thresholding Algorithm """

    def __init__(self):
        
        super(FISTA, self).__init__()

    def set_up(self, initial, b, lam, operator, max_iterations=10, save_interval=1):

        self.lam = lam
        self.operator = operator

        self.b = b

        self.x = initial
        self.y = self.operator.direct(self.x)
        self.y_old = self.y.copy()
        self.w = self.y.copy()

        self.norm = self.operator.norm**2

        self.t=1
        self.t_old = 1

        self.max_iterations = max_iterations
        self.save_interval = save_interval

    def update(self):

        self.y = self.project(self.w)

        self.t = (1 + np.sqrt(1+4*self.t_old**2))/2

        self.w = self.y + (self.t_old-1)/self.t*(self.y-self.y_old)

    def project(self):

        raise NotImplementedError
    

class FISTA_FGP(FISTA):

    def __init__(self):
            
        super(FISTA_FGP, self).__init__()
        
    def project(self, y):
        
        tmp = y - 1/self.norm * self.operator.direct(self.operator.adjoint(y) + self.b)

        return project_inner(tmp, self.lam)
    
class FISTA_TV(FISTA):

    def __init__(self):
            
        super(FISTA_TV, self).__init__()
        
    def project(self, x):
        
        tmp = x - 1/self.norm * self.operator.direct(self.operator.adjoint(x) - self.b)

        return project_inner(tmp, self.lam)
    
from scipy.optimize import minimize, Bounds

class LBFGSB(Algorithm):
    
    def __init__(self):
        
        super(LBFGSB, self).__init__()
        
    def set_up(self, initial, functions, max_iterations=100, save_interval=1, bounds=[0,np.inf]):
    
        self.f = functions

        self.x = initial
        
        self.shape = initial.shape

        self.max_iterations = max_iterations
        self.save_interval = save_interval

        self.bounds = bounds
    def run(self, verbose=0):
        
        obj = self.f(self.x)
        print(obj)
        self.loss.append([obj])
        
        vec = np.reshape(self.x, (self.x.size,))

        bounds = Bounds(np.ones_like(vec)*self.bounds[0], np.ones_like(vec)*self.bounds[1])
        
        lbfgsb = minimize(self.call, vec, method='L-BFGS-B', jac=self.jac, callback=self.call_back,
                          options={'maxiter': self.max_iterations, 'disp': verbose, 'ftol':1e-4}, bounds=bounds)
        
        self.x = np.reshape(lbfgsb.x, self.shape)
        
    def jac(self, x):
        
        im = np.reshape(x, self.shape)        
        
        grad = self.f.gradient(im)
        
        return np.reshape(grad, (grad.size,))
    
    def call(self, x):
        
        im = np.reshape(x, self.shape)
        
        obj =  self.f(im)
        
        self.loss.append([obj])
        
        return obj
    
    def call_back(self, x):
        
        im = np.reshape(x, self.shape)

        obj =  self.f(im)

        self.loss.append([obj])

        print(obj)


from numba import jit

@jit(forceobj=True)
def _armijo(t, x, beta, c, f, f_val, grad_x, max_iters):

    for i in range(max_iters):
        x_new = x - t * grad_x
        f_val_new = f(x_new)
        if f_val_new <= f_val - c * t * (grad_x**2).sum():
            break
        t *= beta
    return t, i

        
class GradientDescent(Algorithm):
    
    def __init__(self, initial, f, step_size=1.0, max_iterations=1000, update_interval=1,
                 armijo_max_iterations = 100,tol=1e-6, armijo_c=0.1, armijo_beta=0.5):
        self.f = f
        self.step_size = step_size  # Initial step size
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.update_interval = update_interval  # Update interval for objective function
        self.tol = tol  # Tolerance for convergence
        self.armijo_c = armijo_c  # Constant for Armijo rule
        self.armijo_beta = armijo_beta  # Reduction factor for step size
        self.armijo_max_iterations = armijo_max_iterations  # Maximum number of iterations for Armijo rule
        self.x = initial.copy()
        self.loss = []
        self.running = True

    def armijo(self):
        t = self.step_size
        f_val = self.f(self.x)
        grad_x = self.f.gradient(self.x)

        return _armijo(t, self.x, self.armijo_beta, self.armijo_c, 
                     self.f, f_val, grad_x, self.armijo_max_iterations)

    def run(self):
            
        self.update_objective(-1)

        for i in range(self.max_iterations):
            
                self.update()

                if self.running:

                    if i % self.update_interval == 0:

                        self.update_objective(i)

                        if self.loss[-2] - self.loss[-1] < self.tol:
                            print("Convergence criterion met")
                            break
                else:
                    break
        
    def update(self):
        step_size, count = self.armijo()

        if count == self.armijo_max_iterations-1:
            print("Armijo rule did not converge")

            self.stopping_criteria()
            
        self.x -= step_size * self.f.gradient(self.x)

        #print(f"Armijo iterations: {count}, step size: {step_size}")

    def update_objective(self, i, verbose = True):

        obj = self.f(self.x)

        if verbose:
            print(f"Iteration {i}: Objective = {obj}")

        self.loss.append(obj)

    @property
    def objective(self):
        return self.loss
        
    
    
    
    