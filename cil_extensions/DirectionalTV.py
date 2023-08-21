from cil.optimisation.operators import LinearOperator, GradientOperator
from cil.framework import BlockGeometry, ImageGeometry
import numpy as np

from numba import jit, njit, prange

@jit(nopython=True)
def array_dot_orig(arr0, arr1):
    out_array = np.zeros_like(arr0[0])
    for i in range(len(arr0)):
        out_array += arr0[i]*arr1[i]
    return out_array

@njit(parallel=True)
def array_dot(arr0, arr1, l):
    out_array = np.zeros(arr0[0].shape)
    for i in prange(l):
        out_array += arr0[i]*arr1[i]
    return out_array
    
def bdc_dot(bdc0, bdc1, image):
    """  """
    arr_list0 = []
    arr_list1 = []
    for i in bdc0.containers:
        arr_list0.append(np.squeeze(i.as_array()))
    for j in bdc1.containers:
        arr_list1.append(np.squeeze(j.as_array()))
    arr = array_dot(np.array(arr_list0),np.array(arr_list1),len(arr_list0)).reshape(image.shape)
    return image.clone().fill(arr)


class DirectionalTV(LinearOperator):
    def __init__(self, anatomical_image, operator, nu = 0.0001, gamma=1, smooth = True, beta = 0.001,**kwargs):
        """Constructor method"""    
        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        self.is2D = False
        self.domain_shape = []
        self.ind = []
        if smooth is True:
            self.beta = beta
        self.voxel_size_order = []
        self._domain_geometry = anatomical_image
        for i, size in enumerate(list(self._domain_geometry.shape) ):
            if size!=1:
                self.domain_shape.append(size)
                self.ind.append(i)
                #self.voxel_size_order.append(self._domain_geometry.spacing[i])
                self.is2D = True

        self.gradient = operator
        
        self.anato = anatomical_image
        self.tmp_im = anatomical_image.clone()
        
        self.gamma = gamma

       	# smoothing for xi 
        self.anato_grad = self.gradient.direct(self.anato) # gradient of anatomical image
        self.denominator = (self.anato_grad.pnorm(2).power(2) + nu**2) #smoothed norm squared of anatomical image  
        self.ndim = len(self.domain_shape)

        super(DirectionalTV, self).__init__(BlockGeometry(*[self._domain_geometry for _ in range(self.ndim)]), 
              range_geometry = BlockGeometry(*[self._domain_geometry for _ in range(self.ndim)]))

    def direct(self, x, out=None): 
        inter_result = bdc_dot(x,self.anato_grad, self.tmp_im)

        if out is None:       
            return x - self.gamma*inter_result*(self.anato_grad/self.denominator)# (delv * delv dot del u) / (norm delv) **2
                
        else:
            out.fill(x - self.gamma*inter_result*(self.anato_grad/self.denominator))
        
    def adjoint(self, x, out=None):  
        return self.direct(x, out=out) # self-adjoint operator