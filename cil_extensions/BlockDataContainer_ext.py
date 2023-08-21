from cil.framework import BlockDataContainer, DataContainer, BlockGeometry
from cil.optimisation.operators import LinearOperator

import numpy as np
from numbers import Number

### Functions to make BlockDataContainer behave like DataContainer

def __new__(cls, *args, **kwargs):
    '''
    Create a new BlockDataContainer object 
    or return a DataContainer if the shape is (1,1)
    '''
    shape = kwargs.get('shape', None)
    if shape is None:
        shape = (len(args),1)
    if shape == (1,1):
        return DataContainer(args[0].as_array(), *args, **kwargs)
    else:
        return super(BlockDataContainer, cls).__new__(cls)
    
def bdc_as_array(self):
    if self.shape == (1,1):
        return self[0].as_array()
    else:
        raise ValueError("Cannot convert BlockDataContainer to array. shape: {}".format(self.shape))
    
def bdc_fill(self, other):
    if isinstance (other, BlockDataContainer):
        if not self.is_compatible(other):
            raise ValueError('Incompatible containers')
        for el,ot in zip(self.containers, other.containers):
            el.fill(ot)

    elif isinstance (other, Number):
        for el in self.containers:
            el.fill(other)
            
    else:
        if self.shape == (1,1):
            self[0].fill(other)
        else:
            raise ValueError('Cannot fill with object provided {}'.format(type(other)))
    
class BDCtoDC(LinearOperator):
    """
    Class to move between DataContainers and BlockDataContainers in CIL
    --- Should probably be moved to src ---
    """
    def __init__(self, image_templ):
        super().__init__(domain_geometry = BlockGeometry(image_templ), range_geometry = image_templ)

    def direct(self, x, out=None):
        if x.shape == (1,1):
            if out is None:
                return x[0]
            else:
                out = x[0]
        else:
            raise ValueError("x is not a 1D BlockDataContainer. shape: {}".format(x.shape))
        
    def adjoint(self, x, out=None):
        if out is None:
            return BlockDataContainer(x)
        else:
            out = BlockDataContainer(x)

    def calculate_norm(self):
        return 1