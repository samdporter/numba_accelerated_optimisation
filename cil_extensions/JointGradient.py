from cil.optimisation.operators import LinearOperator, GradientOperator
from cil.framework import BlockDataContainer, BlockGeometry
import numpy as np


NEUMANN = 'Neumann'
PERIODIC = 'Periodic'
C = 'c'
NUMPY = 'numpy'
CORRELATION_SPACE = "Space"
CORRELATION_SPACECHANNEL = "SpaceChannels"


class JointGradient(LinearOperator):

    def __init__(self, domain_geometries, weightings=None, method='forward', bnd_cond=NEUMANN, **kwargs):
        self.num_modalities = len(domain_geometries)
        self.domain_geometries = domain_geometries
        self.ndims = 2 if domain_geometries[0].shape[0] == 1 else 3

        # Define weightings
        if weightings is None:
            self.weightings = [1] * self.num_modalities
        else:
            assert len(weightings) == self.num_modalities, "Mismatch between number of weightings and domain geometries"

        # Check domain geometries have the same shape
        for idx in range(1, self.num_modalities):
            assert domain_geometries[idx].shape == domain_geometries[0].shape, "All domain geometries must have the same shape"

        self.gradients = [GradientOperator(self.domain_geometries[i], method, bnd_cond, **kwargs) for i in range(self.num_modalities)]

        range_geometry_list = []
        for idx in range(self.num_modalities):
            range_geometry_list += list(self.gradients[idx].range_geometry().allocate().containers)

        range_geometry = BlockGeometry(*range_geometry_list)

        super(JointGradient, self).__init__(domain_geometry=BlockGeometry(*domain_geometries.containers),
                                            range_geometry=range_geometry,)

    def direct(self, x, out=None):
        output = []
        for idx in range(self.num_modalities):
            output += list((self.weightings[idx] * self.gradients[idx].direct(x.containers[idx])).containers)
        return BlockDataContainer(*output)

    def adjoint(self, x, out=None):
        adjoint_inputs = x.containers
        # Split adjoint_inputs into as many lists as there are modalities
        adjoint_inputs = [BlockDataContainer(*adjoint_inputs[i:i + self.ndims]) for i in range(0, len(adjoint_inputs), self.ndims)]

        output = []
        for idx in range(self.num_modalities):
            output.append(self.gradients[idx].adjoint(adjoint_inputs[idx]) / self.weightings[idx])
        return BlockDataContainer(*output)

    def calculate_norm(self):
        # Compute the square root of the sum of squares of norms of each gradient
        return max((self.weightings[idx] * self.gradients[idx].calculate_norm()) ** 2 for idx in range(self.num_modalities))
    
    
class GuidedJointGradient(LinearOperator):
    
    def __init__(self, domain_geometries, fixed_geometries, weightings=None, method='forward', bnd_cond=NEUMANN, **kwargs):
        
        try:
            self.num_fixed = len(fixed_geometries)
        except:
            fixed_geometries = BlockDataContainer(fixed_geometries)
            self.num_fixed = 1
        try:
            self.num_domain = len(domain_geometries)
        except:
            domain_geometries = BlockDataContainer(domain_geometries)
            self.num_domain = 1

        self.num_modalities = self.num_domain + self.num_fixed
        self.domain_geometries = domain_geometries
        self.fixed_geometries = fixed_geometries
        self.ndims = 2 if domain_geometries[0].shape[0] == 1 else 3

        # Define weightings
        if weightings is None:
            self.weightings = [1] * self.num_modalities
        else:
            assert len(weightings) == self.num_modalities, "Mismatch between number of weightings and domain geometries"

        # Check domain geometries have the same shape
        for idx in range(1, self.num_domain):
            assert domain_geometries[idx].shape == domain_geometries[0].shape, "All geometries must have the same shape"
        for idx in range(1, self.num_fixed):
            assert fixed_geometries[idx].shape == domain_geometries[0].shape, "All geometries must have the same shape"

        self.gradients = [GradientOperator(self.domain_geometries[i], method, bnd_cond, **kwargs) for i in range(self.num_domain)]
        self.gradients += [GradientOperator(self.fixed_geometries[i], method, bnd_cond, **kwargs) for i in range(self.num_fixed)]

        range_geometry_list = []
        for idx in range(self.num_modalities):
            range_geometry_list += list(self.gradients[idx].range_geometry().allocate().containers)

        range_geometry = BlockGeometry(*range_geometry_list)

        super(GuidedJointGradient, self).__init__(domain_geometry=self.domain_geometries[0],
                                            range_geometry=range_geometry,)

    def direct(self, x, out=None):
        output = []
        for idx in range(self.num_domain):
            output += list((self.weightings[idx] * self.gradients[idx].direct(x.containers[idx])).containers)
        for idx in range(self.num_fixed):
            output += list((self.weightings[idx + self.num_domain] * self.gradients[idx + self.num_domain].direct(self.fixed_geometries.containers[idx])).containers)
        return BlockDataContainer(*output)

    def adjoint(self, x, out=None):
        adjoint_inputs = x.containers
        # Split adjoint_inputs into as many lists as there are modalities
        adjoint_inputs = [BlockDataContainer(*adjoint_inputs[i:i + self.ndims]) for i in range(0, len(adjoint_inputs), self.ndims)]
        output = []
        for idx in range(self.num_domain):
            output.append(self.gradients[idx].adjoint(adjoint_inputs[idx]) / self.weightings[idx])
        return BlockDataContainer(*output)

    def calculate_norm(self):
        # Compute the square root of the sum of squares of norms of each gradient
        return  max((self.weightings[idx] * self.gradients[idx].calculate_norm()) for idx in range(self.num_domain))
        