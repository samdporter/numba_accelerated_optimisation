from cil.optimisation.operators import LinearOperator
from cil.framework import BlockDataContainer, BlockGeometry

class BDC_to_DC(LinearOperator):

    def __init__(self, geometry):
        range_geometry = geometry
        super(BDC_to_DC, self).__init__(domain_geometry = BlockDataContainer(geometry), 
                                        range_geometry = range_geometry)

    def direct(self, x, out=None):
        return x[0]
    
    def adjoint(self, x, out=None):
        return BlockDataContainer(x)
    
    def calculate_norm(self):
        return 1