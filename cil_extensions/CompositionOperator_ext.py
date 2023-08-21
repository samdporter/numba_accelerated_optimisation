from cil.optimisation.operators import LinearOperator
import numpy as np
import warnings

def CO_calculate_norm(self, **kwargs):
    '''Returns an estimate of the norm of the Composition Operator

    if the operator in the block do not have method norm defined, i.e. they are SIRF
    AcquisitionModel's we use PowerMethod if applicable, otherwise we raise an Error
    '''
    for op in self.operators:
        if not op.is_linear():
            raise NotImplementedError('The norm of the composition operator is only defined for linear operators')
        
    return LinearOperator.PowerMethod(self)
