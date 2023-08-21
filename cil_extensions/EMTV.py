import logging

import numpy as np
from cil.optimisation.algorithms import FISTA, OPDHG, PDHG, SPDHG, Algorithm, GD
from cil.optimisation.functions import (BlockFunction, IndicatorBox,
                                        L2NormSquared, WeightedL2NormSquared,
                                        OperatorCompositionFunction,
                                        Function, KullbackLeibler,)
from cil.optimisation.operators import (BlockOperator, IdentityOperator,
                                        ZeroOperator, LinearOperator,
                                        CompositionOperator, Operator,)
from sirf.STIR import (MessageRedirector, TruncateToCylinderProcessor,
                       ImageData, ObjectiveFunction,)
from cil.framework import BlockDataContainer, DataContainer

from cil_extensions.MAPEM import MAPEM
from cil_extensions.PD3O import PD3O
from cil_extensions.BDC_Indicator import IndicatorBox as BDC_IndicatorBox
from cil_extensions.misc import BDC_to_DC
from cil_extensions.PD3O import PD3O

from scipy.optimize import minimize, Bounds
from scipy.optimize._lbfgsb_py import _minimize_lbfgsb as bfgs

class EMTV_old(Algorithm):

    r""" 
    Class for the EMTV algorithm for Total Variation minimisation.
    Inner iterations are performed using the PD3O algorithm.
    """
    

    def __init__(self, initial, data_fidelities, prior, operator = None, 
                 algorithm = 'pdhg', num_subsets=1, omega = 0.5, eps = 1e-8, 
                 max_inner_iter=50, mapem=False, svrg=False, **kwargs):
        
        super(EMTV_old, self).__init__(**kwargs)

        self.x = initial.copy()
        self.z = self.x.copy()
        self.s = self.x.copy()

        self.eps = eps
        self.omega = omega

        self.data_fidelities = data_fidelities
        self.prior = prior
        self.operator = operator
        self.algorithm = algorithm.lower()

        self.num_subsets = num_subsets
        self.subset = 0

        self.max_inner_iter = max_inner_iter

        self.cyl = TruncateToCylinderProcessor()

        self.mapem = mapem
        self.svrg = svrg

        self.subset_list = []

        if self.svrg:
            self.full_gradient = self.x.get_uniform_copy(0)
            self.grads = [self.x.get_uniform_copy(0) for _ in range(self.num_subsets)]

        self.bo = BlockOperator(IdentityOperator(self.x), self.operator)

        self.set_up()

    def set_up(self, *args, **kwargs):
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
    
    def update(self, *args, **kwargs):

        if self.svrg:
            self.subset = np.random.randint(0, self.num_subsets)
            if self.iteration % self.num_subsets == 0:
                self.full_gradient, self.grads = self.calculate_gradients()

        if len(self.subset_list) < self.num_subsets:
            self.subset_list.append(self.data_fidelities.get_subset_sensitivity(self.subset).maximum(0)+self.eps)
            self.s.fill(self.subset_list[-1])
            print("calculating subset {}".format(self.subset))
        else:
            self.s.fill(self.subset_list[self.subset])

        self.osem_step()

        self.denoise_step()

        self.subset = (self.subset+1)%self.num_subsets

    def osem_step(self):
        ratio = self.data_fidelities.get_backprojection_of_acquisition_ratio(self.x, self.subset).maximum(0)
        self.z.fill(self.divide((self.x.multiply(ratio)),self.s) ) 
        self.cyl.apply(self.z)
    
    def denoise_step(self):

        if self.mapem:
            f = MAPEM(lam=self.z, s=self.s)

        else:
            w = self.divide(self.s, self.x)+self.eps
            f = WeightedL2NormSquared(weight=w, b=self.z)

        if self.algorithm == 'spdhg':
            ii = 2*self.max_inner_iter
            probs = [0.5,0.5]
            bf = BlockFunction(f, self.prior)
            inner_solver = SPDHG(initial=self.z, f=bf, g=BDC_IndicatorBox(lower=0), operator=self.bo, 
                                max_iteration=ii, prob=probs, update_objective_interval=ii//2)
        elif self.algorithm == 'opdhg':
            ii = 2*self.max_inner_iter
            bf = BlockFunction(f, self.prior)
            inner_solver = OPDHG(initial=self.z, f=bf, g=BDC_IndicatorBox(lower=0), operator=self.bo, 
                                max_iteration=ii, gamma = 1,
                                update_objective_interval=ii//2)
        elif self.algorithm == 'pdhg':
            inner_solver = PDHG(initial=self.z, f=self.prior, g = f , 
                                operator=self.operator, max_iteration=self.max_inner_iter,
                                update_objective_interval=self.max_inner_iter//2)
        elif self.algorithm == 'pd30':
            sigma = 2/f.L
            tau = 1/(self.norm**2*sigma)
            inner_solver = PD3O(f, BDC_IndicatorBox(lower=0), self.prior, self.operator, 
                                initial=self.z,  max_iteration=self.max_inner_iter,
                                sigma=sigma, tau=tau, update_objective_interval=self.max_inner_iter//2)
        else:
            raise ValueError('Algorithm must be either pdhg, fista or pd30')
        
        inner_solver.run(verbose=2)
        self.x.fill(inner_solver.solution.maximum(0))

        del inner_solver, f

        try:
            del bf
        except:
            pass

    def calculate_gradients(self):
        raise NotImplementedError

    def update_objective(self):
        df  = -self.data_fidelities(self.x)
        p = self.prior(self.operator.direct(self.x))
        objective = df + p

        self.loss.append([objective, df, p])

    @property
    def objective(self):
         '''alias of loss'''
         return [x for x in self.loss]

    def divide(self, a,b):
        return a.divide(b+self.eps)
    





class EMTV(Algorithm):
    def __init__(self, initial, data_fidelities, prior, operator=None, algorithm='pdhg', num_subsets=None, omega=0.5, 
                 eps=1e-8, max_inner_iter=50, inner_func='kl', svrg=False, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps # epsilon for division
        self.rho = 0.99
        self.omega = omega # relaxation parameter
        self.data_fidelities = data_fidelities # data fidelity function(s)
        self.prior = prior # prior function(s)
        self.operator = operator # operator(s)
        self.algorithm = algorithm.lower() # algorithm to use for denoising step

        if num_subsets is None:
            if isinstance(data_fidelities, BlockFunction):
                num_subsets = [1] * len(data_fidelities.functions)
            else:
                num_subsets = [1]
        else:
            if isinstance(data_fidelities, BlockFunction):
                assert len(num_subsets) == len(data_fidelities.functions), 'Number of subsets must be equal to number of data_fidelities'
                self.num_subsets = num_subsets # number of subsets for OSEM step
            else:
                if isinstance(num_subsets, list):
                    assert len(num_subsets) == 1, 'Number of subsets must be equal to 1'
                    self.num_subsets = num_subsets
                elif isinstance(num_subsets, int):
                    self.num_subsets = [num_subsets]#
                else:
                    raise ValueError('num_subsets must be either a list or an integer')

        if isinstance(data_fidelities, BlockFunction):
            self.subset = [0] * len(data_fidelities.functions) # current subset for OSEM step
        else:
            self.subset = [0]
        self.max_inner_iter = max_inner_iter # maximum number of iterations for denoising step
        self.cyl = TruncateToCylinderProcessor() # truncate to cylinder processor tp remove edge effects
        self.inner_func = inner_func # use MAPEM or for denoising step (True/False)
        self.svrg = svrg # use SVRG for OSEM step (True/False)
        self.x = initial.copy() # image estimate
        self.s = self.x.copy() # sensitivity image
        self.z = self.x.copy() # intermediate image estimate 

        if self.svrg:
            self.prepare_for_svrg()
            
        self.operator
        if isinstance(self.x, BlockDataContainer):
            ids = [IdentityOperator(x) for x in self.x.containers]
        elif isinstance(self.x, (ImageData, DataContainer)):
            ids = [IdentityOperator(self.x)]
        self.bo = BlockOperator(*ids, self.operator) # block operator for denoising step
        self.set_up()

    def build_bdc_operator(self):
        if isinstance(self.x, BlockDataContainer):
            return BlockOperator(*[BDC_to_DC(x) for x in self.x.containers])
        if isinstance(self.x, (DataContainer, ImageData)):
            return IdentityOperator(self.x)
        raise ValueError('x must be a DataContainer or a BlockDataContainer')

    def prepare_for_svrg(self):
        # not currently used
        self.full_gradient = self.x.get_uniform_copy(0)
        self.grads = [self.x.get_uniform_copy(0) for _ in range(self.num_subsets)]

    def set_up(self, *args, **kwargs):
        self.configured = True
        logging.info(f"{self.__class__.__name__} configured")

    def update(self, *args, **kwargs):
        if self.svrg and self.iteration % self.num_subsets == 0:
            self.full_gradient, self.grads = self.calculate_gradients()

        self.get_sensitivities()
        self.osem_step()

        if isinstance(self.x, BlockDataContainer):
            [self.cyl.apply(z) for z in self.z.containers]
        else:
            self.cyl.apply(self.z)

        self.denoise_step()
        self.subset = [(self.subset[i] + 1) % self.num_subsets[i] for i in range(len(self.num_subsets))]

    def get_sensitivities(self):
        
        if isinstance(self.data_fidelities, BlockFunction):
            if isinstance(self.data_fidelities.functions[0], ObjectiveFunction):
                [el.fill(df.get_subset_sensitivity(self.subset[i])) for i, (el, df) in enumerate(zip(self.s.containers, self.data_fidelities))]
            elif isinstance(self.data_fidelities.functions[0], KullbackLeibler):
                [el.fill(df.operator.adjoint(df.operator.range_geometry.allocate(1))) for i, (el, df) in enumerate(zip(self.s.containers, self.data_fidelities))]
        
        elif isinstance(self.data_fidelities, (ObjectiveFunction)):
            self.s.fill(self.data_fidelities.get_subset_sensitivity(self.subset[0]))
        
        elif isinstance(self.data_fidelities, KullbackLeibler):
            self.s.fill(self.data_fidelities.operator.adjoint(self.data_fidelities.operator.range_geometry.allocate(1)))
        
        else:
            raise ValueError("Data fidelity must be either a BlockFunction or Function")
            
    def osem_step(self):
        if isinstance(self.x, BlockDataContainer):
            if isinstance(self.data_fidelities.functions[0], ObjectiveFunction):
                for i, (el, df, im, s) in enumerate(zip(self.z.containers, self.data_fidelities.functions, self.x.containers, self.s.containers)):
                    ratio = df.get_backprojection_of_acquisition_ratio(im, self.subset[i]).maximum(0)
                    el.fill(self.divide(im.multiply(ratio), s))
    
            elif isinstance(self.data_fidelities.functions[0], KullbackLeibler):
                for el, df, im, s in zip(self.z.containers, self.data_fidelities.functions, self.x.containers, self.s.containers):
                    m = self.divide(im, s)
                    el.fill((el - m * df.gradient(el)).maximum(0))
            else:
                raise ValueError("Data fidelity must be either a KullbackLeibler or ObjectiveFunction")
        elif isinstance(self.data_fidelities, ObjectiveFunction):

            ### TODO: check this for possible x-y flip ###
            ### Does this need manually flipping if SPECT? ###
            ratio = self.data_fidelities.get_backprojection_of_acquisition_ratio(self.x, self.subset[0]).maximum(0)
            self.z.fill(self.divide(self.x.multiply(ratio), self.s))

        elif isinstance(self.data_fidelities, KullbackLeibler):

            m = self.divide(self.x, self.s)
            self.x.fill((self.x - m * self.data_fidelities.gradient(self.x)).maximum(0))
        else:
            raise ValueError("Data fidelity must be either a KullbackLeibler or ObjectiveFunction")

    def create_denoising_fidelity(self):
        """
        Create the function list based on the MAPEM flag.
        """
        if self.inner_func.lower() == 'mapem':
            if isinstance(self.x, BlockDataContainer):
                return BlockFunction(*[MAPEM(lam=z, s=s) for z, s in 
                                       zip(self.z.containers, self.s.containers)])
            elif isinstance(self.x, (ImageData, DataContainer)):
                return MAPEM(lam=self.z, s=self.s)
        elif self.inner_func.lower() == 'wl2norm':
            if isinstance(self.x, BlockDataContainer):
                return  BlockFunction(*[self.create_weighted_L2NormSquared(x, z, s) 
                                        for x, z, s in zip(self.x.containers, self.z.containers, 
                                                           self.s.containers)])
            elif isinstance(self.x, (ImageData, DataContainer)):
                return self.create_weighted_L2NormSquared(self.x, self.z, self.s)
        elif self.inner_func.lower() == 'kl':
            if isinstance(self.x, BlockDataContainer):
                return  BlockFunction(*[KullbackLeibler(b=z) for z in self.z.containers])
            elif isinstance(self.x, (ImageData, DataContainer)):
                return KullbackLeibler(b=self.z)
        elif self.inner_func.lower() == 'l2norm':
            if isinstance(self.x, BlockDataContainer):
                return  BlockFunction(*[L2NormSquared(b=z) for z in self.z.containers])
            elif isinstance(self.x, (ImageData, DataContainer)):
                return L2NormSquared(b=self.z)

    def create_weighted_L2NormSquared(self, x, z, s):
        """
        Create a WeightedL2NormSquared function.
        """
        w = self.divide(s, x) + self.eps
        return WeightedL2NormSquared(weight=w, b=z)

    def denoise_step(self):
        """
        Determine which algorithm to use based on the input string.
        """
        primal_funcs = self.create_denoising_fidelity()

        if self.algorithm == 'spdhg':
            self.run_spdhg(primal_funcs, self.max_inner_iter)
        elif self.algorithm == 'opdhg':
            self.run_opdhg(primal_funcs, self.max_inner_iter)
        elif self.algorithm == 'pdhg':
            self.run_pdhg(primal_funcs, self.max_inner_iter)
        elif self.algorithm == 'gd':
            self.run_gd(primal_funcs, self.max_inner_iter)
        elif self.algorithm == 'pd30':
            self.run_pd30(primal_funcs, self.max_inner_iter)
        else:
            raise ValueError('Algorithm must be either "spdhg", "opdhg", or "gd". Your input was: {}'.format(self.algorithm))   
        
    def run_spdhg(self, pf, ii):

        if isinstance(pf, BlockFunction):
            f = BlockFunction(*pf.functions, self.prior)
            
        else:
            f = BlockFunction(pf, self.prior)
        probs = [1/len(f.functions) for _ in f.functions]
        gamma = 1
        tau = self.rho
        sigma = [self.rho / self.operator.norm()**2 for _ in f.functions]
        inner_solver = SPDHG(initial=self.z, f=f, g=BDC_IndicatorBox(0), operator=self.bo, 
                            max_iteration=ii, prob=probs, update_objective_interval=10, rho=0.99, 
                            tau = tau, sigma = sigma)
        inner_solver.run(verbose=2)
        self.x = inner_solver.solution.maximum(0)
        del inner_solver
    def run_opdhg(self, pf, ii):
        if isinstance(pf, BlockFunction):
            f = BlockFunction(*pf.functions, self.prior)
        else:
            f = BlockFunction(pf, self.prior)
        tau = self.rho/self.operator.norm()**2
        sigma = [tau / self.operator.norm()**2 for _ in f.functions]
        inner_solver = OPDHG(initial=self.z, f=f, g=BDC_IndicatorBox(0), operator=self.bo, 
                            max_iteration=ii, update_objective_interval=100, rho=0.99,
                            tau = tau, sigma = sigma)
        inner_solver.run(verbose=2)
        self.x = inner_solver.solution.maximum(0)
        del inner_solver

    def run_pdhg(self, pf, ii):
        gamma = 1
        tau = gamma * self.rho
        sigma = gamma * self.rho / self.operator.norm()**2
        inner_solver = PDHG(initial=self.z, f=self.prior, g=pf, operator=self.operator, 
                            max_iteration=ii, update_objective_interval=10, tau=tau, sigma=sigma)
        inner_solver.run(verbose=2)
        self.x = inner_solver.solution.maximum(0)
        del inner_solver

    def run_pd30(self, pf, ii):
        rho = 0.99
        try:
            tau = 2 / np.max([f.L for f in pf.containers])
        except:
            tau = 2 / pf.L
        sigma = tau / self.operator.norm()**2
        inner_solver = PD3O(initial=self.z, f=pf, g = BDC_IndicatorBox(lower=0), h=self.prior, 
                            operator=self.operator, max_iteration=ii, update_objective_interval=100,
                            tau=rho*tau, sigma=rho*sigma)
        inner_solver.run(verbose=2)
        self.x = inner_solver.solution.maximum(0)
        del inner_solver

    def run_gd(self, pf, ii):
        if isinstance(self.prior, BlockFunction):
            df = self.prior.functions
        else:
            df = self.prior

        gd = GD(initial=self.z+self.eps, objective_function=pf+df, step_size=0.0001, max_iteration=100, update_objective_interval=1)#ii//2)
        gd.run(iterations=100)
        self.x = gd.solution.maximum(0)
        del gd

    def calculate_gradients(self):
        raise NotImplementedError

    def update_objective(self):
        dl = []
        
        ### Data fidelity ###
        if isinstance(self.x, BlockDataContainer):
            # multiple modalities
            for df, x in zip(self.data_fidelities, self.x.containers):
                dl.append(-df(x))
        elif isinstance(self.x, (ImageData, DataContainer)):
            # single modality
            dl.append(-self.data_fidelities(self.x))
    
        ### Prior ###
        if isinstance(self.prior, BlockFunction) and isinstance(self.operator, BlockOperator):
            # multiple priors
            for p, o in zip(self.prior.functions, self.operator.operators):
                dl.append(p(o.direct(self.x)))
        elif isinstance(self.prior, Function) and isinstance(self.operator, Operator):
            # single prior
            dl.append(self.prior(self.operator.direct(self.x)))
        else:
            raise ValueError("Prior must be either a BlockFunction or Function and operator must be either a BlockOperator or Operator")
        
        self.loss.append([sum(dl)]+dl)

    @property
    def objective(self):
         '''alias of loss'''
         return [x for x in self.loss]

    def divide(self, a,b):
        return a.divide(b+self.eps)