import numpy as np
from numba import njit, prange

from python_optimisation.misc import power_iteration

from skimage.transform import resize



######################
## Operator Base #####
######################

class Operator():

    def __init__():
        raise NotImplementedError

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError
    
    def adjoint(self, x):
        return self.backward(x)
    
    def direct(self, x):
        return self.forward(x)
    
    def calculate_norm(self, x):
        if not hasattr(self, 'norm'):
            self.norm = power_iteration(self, x)
            return self.norm
        else:
            return self.norm

    
class CompositionOperator(Operator):
    
    def __init__(self, ops):
        self.ops = ops
        
    def direct(self, x):
        res = x.copy()
        for op in self.ops:
            res = op.direct(res)
        return res
    
    def adjoint(self,x):
        res = x.copy()
        for op in self.ops[::-1]:
            res = op.adjoint(res)
        return res
    
    def forward(self, x):
        return self.direct(x)
    
    def backward(self, x):
        return self.adjoint(x)
    
class BlockOperator(Operator):

    def __init__(self, operators):
        self.operators = operators

    def direct(self, x):
        res = []
        x = np.moveaxis(x, -1, 0)
        for op, arr in zip(self.operators,x):
            res.append(op.direct(arr))
        return np.moveaxis(np.array(res), 0, -1)
    
    def adjoint(self, x):
        res = []
        x = np.moveaxis(x, -1, 0)
        for op, arr in zip(self.operators,x):
            res.append(op.adjoint(arr))
        return np.moveaxis(np.array(res), 0, -1) 
    
    def forward(self, x):
        return self.direct(x)
    
    def backward(self, x):
        return self.adjoint(x)   
    
######################################
###### Operator Classes ##############
######################################

class Resample(Operator):

    def __init__(self, new_shape, old_shape, order=1, mode='reflect', cval=0):
        self.new_shape = new_shape
        self.old_shape = old_shape
        self.order = order
        self.mode = mode
        self.cval = cval
    def forward(self, x):
        return resize(x, self.new_shape, order=self.order, 
                      mode=self.mode, cval=self.cval)
    
    def backward(self, x):
        return resize(x, self.old_shape, order=self.order, 
                      mode=self.mode, cval=self.cval, 
                      prefilter=self.prefilter)
    
    def direct(self, x):
        return self.forward(x)
    
    def adjoint(self, x):
        return self.backward(x)


class IdentityOperator(Operator):

    def __init__(self):
        self.norm = 1

    def forward(self, x):
        return x

    def backward(self, x):
        return x
    
    def direct(self, x):
        return x
    
    def adjoint(self, x):
        return x

######################
## Gradient ##########
######################

class Gradient(Operator):
    
    def __init__(self, voxel_size=(1,1,1), edge_strategy='Periodic', direction='forward') -> None:
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size

    def forward(self, x):
        if self.direction == 'forward':
            return forward_difference_3d(x, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            return backward_difference_3d(x, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def backward(self, x):
        if self.direction == 'forward':
            return backward_difference_transpose_3d(x, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            return forward_difference_transpose_3d(x, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def direct(self,x):
        return self.forward(x)
    
    def adjoint(self,x):
        return self.backward(x)
    
class JointGradient(Operator):
    
    def __init__(self, num_modalities, ndims, voxel_size=(1,1,1), edge_strategy='replication', direction='forward') -> None:
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size
        self.num_modalities = num_modalities
        self.ndims = ndims

    def forward(self, x):
        # split images into list of arrays - remove singleton final dimension
        images = np.split(x, x.shape[-1], axis=-1)
        images = [np.squeeze(image) for image in images]
        # compute gradient for each image
        gradients = []
        for image in images:
            if self.direction == 'forward':
                image_grad = forward_difference_3d(image, self.voxel_size, self.edge_strategy)
                # split into list of arrays - remove singleton final dimension
            elif self.direction == 'backward':
                gradients.append(backward_difference_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
            image_grads = np.split(image_grad, self.ndims, axis=-1)
            image_grads = [np.squeeze(image_grad) for image_grad in image_grads]
            gradients+=image_grads
        # stack along axis to produce the desired shape (z, y, x, image, num_modalities*ndims)
        return np.stack(gradients, axis=-1)

    def backward(self, x):
        images = np.split(x, x.shape[-1] // self.ndims, axis=-1)
        out = []
        for image in images:
            if self.direction == 'forward':
                out.append(backward_difference_transpose_3d(image, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                out.append(forward_difference_transpose_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
        return np.stack(out, axis=-1)
    
class JointGradientNew(Operator):

    def __init__(self, num_modalities, ndims, voxel_size=(1,1,1), edge_strategy='Periodic', direction='forward') -> None:
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size
        self.num_modalities = num_modalities
        self.ndims = ndims

    def forward(self, x):
        # split images into list of arrays - remove singleton final dimension
        images = np.split(x, x.shape[-1], axis=-1)
        images = [np.squeeze(image) for image in images]
        # compute gradient for each image
        gradients = []
        for image in images:
            if self.direction == 'forward':
                image_grad = forward_difference_3d(image, self.voxel_size, self.edge_strategy)
                # split into list of arrays - remove singleton final dimension
            elif self.direction == 'backward':
                image_grad = backward_difference_3d(image, self.voxel_size, self.edge_strategy)
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
            image_grads = np.split(image_grad, self.ndims, axis=-1)
            image_grads = [np.squeeze(image_grad) for image_grad in image_grads]
            gradients+=image_grads
        # stack along axis to produce the desired shape (z, y, x, image, num_modalities*ndims)
        return np.stack(gradients, axis=-1)

    
    def backward(self, x):
        images = np.split(x, x.shape[-1] // self.ndims, axis=-1)
        out = []
        for image in images:
            if self.direction == 'forward':
                out.append(backward_difference_transpose_3d(image, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                out.append(forward_difference_transpose_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
        return np.stack(out, axis=-1)
    
    def direct(self, x):
        return self.forward(x)
    
    def adjoint(self, x):
        return self.backward(x)
    
class DirectionalJointGradient(Operator):
    
    def __init__(self, anatomical, num_modalities, ndims, gamma=1, eta=1e-6,
                 voxel_size=(1,1,1), edge_strategy='Periodic', direction='forward') -> None:
        
        self.gamma = gamma
        self.eta = eta
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size
        self.num_modalities = num_modalities
        self.ndims = ndims
        
        self.anatomical = anatomical.astype(np.float64)
        if self.direction == 'forward':
            self.anat_grad = forward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            self.anat_grad = backward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def forward(self, x):
        # split images into list of arrays - remove singleton final dimension
        images = np.split(x, x.shape[-1], axis=-1)
        images = [np.squeeze(image) for image in images]
        # compute gradient for each image
        gradients = []
        for image in images:
            if self.direction == 'forward':
                image_grad = directional_op(forward_difference_3d(image, self.voxel_size, self.edge_strategy), 
                                                   self.anat_grad, self.gamma, self.eta)
                # split into list of arrays - remove singleton final dimension
                image_grads = np.split(image_grad, self.ndims, axis=-1)
                image_grads = [np.squeeze(image_grad) for image_grad in image_grads]
                gradients+=image_grads
            elif self.direction == 'backward':
                gradients.append(backward_difference_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
        # stack along axis to produce the desired shape (z, y, x, image, num_modalities*ndims)
        return np.stack(gradients, axis=-1)

    
    def backward(self, x):
        images = np.split(x, x.shape[-1] // self.ndims, axis=-1)
        out = []
        for image in images:
            if self.direction == 'forward':
                out.append(backward_difference_transpose_3d(directional_op(image, self.anat_grad, self.gamma, self.eta),
                                                              self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                out.append(forward_difference_transpose_3d(directional_op(image, self.anat_grad, self.gamma, self.eta),
                                                                                self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))
        return np.stack(out, axis=-1)
    
    def direct(self, x):
        return self.forward(x)
    
    def adjoint(self, x):
        return self.backward(x)


######################
## Directional #######
######################

class DirectionalGradient(Operator):

    """
    Directional operator for edge-preserving smoothing

    """

    def __init__(self, anatomical, gamma=1, eta=1e-6,
                 voxel_size=(1,1,1), edge_strategy='Periodic', direction='forward') -> None:
        
        self.gamma = gamma # edge preservation parameter
        self.eta = eta # regularisation parameter
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size
            
        self.anatomical = anatomical.astype(np.float64)
        if self.direction == 'forward':
            self.anat_grad = forward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            self.anat_grad = backward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def forward(self, x):
        
        if self.direction == 'forward':
            res = forward_difference_3d(x, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            res = backward_difference_3d(x, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
        return directional_op(res, self.anat_grad, self.gamma, self.eta)
        
    def backward(self, x):
        
        tmp = directional_op(x, self.anat_grad, self.gamma, self.eta)
        
        if self.direction == 'forward':
            return backward_difference_transpose_3d(tmp, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            return forward_difference_transpose_3d(tmp, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def direct(self,x):
        return self.forward(x)
    
    def adjoint(self,x):
        return self.backward(x)

######################
## Jacobian ##########
######################

class Jacobian(Operator):
    def __init__(self, edge_strategy='Periodic', voxel_size=(1,1,1), direction='forward'):
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size

    def __call__(self, images):
        return self.direct(images)

    def direct(self, images):
        # Number of images in the array
        num_images = images.shape[-1]

        # List to store the computed Jacobians
        jac_list = []

        for idx in range(num_images):
            image = images[..., idx]
            if self.direction == 'forward':
                jac_list.append(forward_difference_3d(image, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                jac_list.append(backward_difference_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))

        # Stack along the new axis to produce the desired shape (z, y, x, image, dz, dy, dx)
        return np.stack(jac_list, axis=-2)

    def adjoint(self, jacobians):
        # Number of jacobians in the array (matches number of images)
        num_jacobians = jacobians.shape[-2]

        # List to store the computed adjoints
        adjoint_list = []

        for idx in range(num_jacobians):
            jacobian = jacobians[..., idx, :]
            if self.direction == 'forward':
                adjoint_list.append(backward_difference_transpose_3d(jacobian, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                adjoint_list.append(forward_difference_transpose_3d(jacobian, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))

        # Stack along the image axis to produce the desired shape (z, y, x, image)
        return np.stack(adjoint_list, axis=-1)
    
    def forward(self, images):
        return self.direct(images)
    
    def backward(self, jacobians):
        return self.adjoint(jacobians)
    
class DirectionalJacobian(Operator):
    
    def __init__(self, anatomical, gamma=1, eta=1e-6,
                 voxel_size=(1,1,1), edge_strategy='Periodic', direction='forward') -> None:
        
        self.gamma = gamma # edge preservation parameter
        self.eta = eta # regularisation parameter
        
        self.edge_strategy = edge_strategy
        self.direction = direction
        self.voxel_size = voxel_size
            
        self.anatomical = anatomical.astype(np.float64)
        if self.direction == 'forward':
            self.anat_grad = forward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        elif self.direction == 'backward':
            self.anat_grad = backward_difference_3d(anatomical, self.voxel_size, self.edge_strategy)
        else:
            raise ValueError('Invalid direction: {}'.format(self.direction))
        
    def __call__(self, images):
        return self.direct(images)

    def direct(self, images):
        # Number of images in the array
        num_images = images.shape[-1]

        # List to store the computed Jacobians
        jac_list = []

        for idx in range(num_images):
            image = images[..., idx]
            if self.direction == 'forward':
                jac_list.append(forward_difference_3d(image, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                jac_list.append(backward_difference_3d(image, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))

        # Stack along the new axis to produce the desired shape (z, y, x, image, dz, dy, dx)
        
        jac_list = [directional_op(jac, self.anat_grad, self.gamma, self.eta) for jac in jac_list]
        
        return np.stack(jac_list, axis=-2)

    def adjoint(self, jacobians):
        # Number of jacobians in the array (matches number of images)
        num_jacobians = jacobians.shape[-2]

        # List to store the computed adjoints
        adjoint_list = []

        for idx in range(num_jacobians):
            jacobian = directional_op(jacobians[..., idx, :], self.anat_grad, self.gamma, self.eta)
            if self.direction == 'forward':
                adjoint_list.append(backward_difference_transpose_3d(jacobian, self.voxel_size, self.edge_strategy))
            elif self.direction == 'backward':
                adjoint_list.append(forward_difference_transpose_3d(jacobian, self.voxel_size, self.edge_strategy))
            else:
                raise ValueError('Invalid direction: {}'.format(self.direction))

        # Stack along the image axis to produce the desired shape (z, y, x, image)
        return np.stack(adjoint_list, axis=-1)
    
    def forward(self, images):
        return self.direct(images)
    
    def backward(self, jacobians):
        return self.adjoint(jacobians)
        
        
        
        

    
######################
## Identity #########
######################

class Identity(Operator):

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x

    def forward(self, x):
        return x

    def backward(self, x):
        return x
    
######################
## Wavelets ##########
######################

import pywt

### How can I multiply in the wavelet do0main by an array? Does this even make sense? ###

class WaveletTransform(Operator):

    """ change this to return a wavelet transform object"""

    def __init__(self, wavelet='db2', levels=2) -> None:
        self.levels = levels
        self.wavelet = wavelet

    def forward(self, x):
        # discrete n-level nd wavelet transform
        return pywt.wavedecn(x, self.wavelet, level=self.levels)
    
    def backward(self, x):
        # inverse discrete n-level nd wavelet transform
        return pywt.waverecn(x, self.wavelet)
    
    def direct(self, x):
        return self.forward(x)
    
    def adjoint(self, x):
        return self.backward(x)
    
    def __call__(self, x):
        return self.forward(x)

########################
## Finite Differences ##
########################

@njit(parallel=True)
def backward_difference_3d(image, voxel_size=(1,1,1), edge_strategy='Neumann'):
    
    dx, dy, dz = voxel_size
    
    depth, height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    grad_z = np.zeros_like(image)

    for d in prange(1, depth):
        grad_x[d, :, :] = (image[d, :, :] - image[d - 1, :, :]) / dx
    for h in prange(1, height):
        grad_y[:, h, :] = (image[:, h, :] - image[:, h - 1, :]) / dy
    for w in prange(1, width):
        grad_z[:, :, w] = (image[:, :, w] - image[:, :, w - 1]) / dz

    if edge_strategy == 'Neumann':
        grad_x[0, :, :] = 0
        grad_y[:, 0, :] = 0
        grad_z[:, :, 0] = 0
    elif edge_strategy == 'Periodic':
        grad_x[0, :, :] = (image[0, :, :] - image[-1, :, :]) / dx
        grad_y[:, 0, :] = (image[:, 0, :] - image[:, -1, :]) / dy
        grad_z[:, :, 0] = (image[:, :, 0] - image[:, :, -1]) / dz

    gradient = np.stack((grad_x, grad_y, grad_z), axis=-1)

    return gradient

@njit(parallel=True)
def backward_difference_transpose_3d(gradient, voxel_size=(1,1,1), edge_strategy='Neumann'):
    dx, dy, dz = voxel_size
    depth, height, width = gradient.shape[:-1]  # Excluding the gradient dimension
    img_x = np.zeros((depth, height, width))
    img_y = np.zeros((depth, height, width))
    img_z = np.zeros((depth, height, width))

    grad_x, grad_y, grad_z = gradient[..., 0], gradient[..., 1], gradient[..., 2]

    for d in prange(1, depth):
        img_x[d, :, :] = (grad_x[d-1, :, :] - grad_x[d, :, :]) / dx
    for h in prange(1, height):
        img_y[:, h, :] = (grad_y[:, h-1, :] - grad_y[:, h, :]) / dy
    for w in prange(1, width):
        img_z[:, :, w] = (grad_z[:, :, w-1] - grad_z[:, :, w]) / dz

    if edge_strategy == 'Neumann':
        img_x[0, :, :] = 0
        img_y[:, 0, :] = 0
        img_z[:, :, 0] = 0
    elif edge_strategy == 'Periodic':
        img_x[0, :, :] = (grad_x[-1, :, :] - grad_x[0, :, :]) / dx
        img_y[:, 0, :] = (grad_y[:, -1, :] - grad_y[:, 0, :]) / dy
        img_z[:, :, 0] = (grad_z[:, :, -1] - grad_z[:, :, 0]) / dz

    image = img_x + img_y + img_z
    return image


@njit(parallel=True)
def forward_difference_3d(image, voxel_size=(1,1,1), edge_strategy='Neumann'):
    dx, dy, dz = voxel_size
    depth, height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    grad_z = np.zeros_like(image)

    for d in prange(depth - 1):
        grad_x[d, :, :] = (image[d + 1, :, :] - image[d, :, :]) / dx
    for h in prange(height - 1):
        grad_y[:, h, :] = (image[:, h + 1, :] - image[:, h, :]) / dy
    for w in prange(width - 1):
        grad_z[:, :, w] = (image[:, :, w + 1] - image[:, :, w]) / dz

    if edge_strategy == 'Neumann':
        grad_x[-1, :, :] = 0
        grad_y[:, -1, :] = 0
        grad_z[:, :, -1] = 0
    elif edge_strategy == 'Periodic':
        grad_x[-1, :, :] = (image[0, :, :] - image[-1, :, :]) / dx
        grad_y[:, -1, :] = (image[:, 0, :] - image[:, -1, :]) / dy
        grad_z[:, :, -1] = (image[:, :, 0] - image[:, :, -1]) / dz

    gradient = np.stack((grad_x, grad_y, grad_z), axis=-1)
    return gradient


@njit(parallel=True)
def forward_difference_transpose_3d(gradient, voxel_size=(1,1,1), edge_strategy='Neumann'):
    dx, dy, dz = voxel_size
    depth, height, width = gradient.shape[:-1]  # Excluding the gradient dimension
    img_x = np.zeros((depth, height, width))
    img_y = np.zeros((depth, height, width))
    img_z = np.zeros((depth, height, width))

    grad_x, grad_y, grad_z = gradient[..., 0], gradient[..., 1], gradient[..., 2]

    for d in prange(depth - 1):
        img_x[d, :, :] = -(grad_x[d+1, :, :] - grad_x[d, :, :]) / dx
    for h in prange(height - 1):
        img_y[:, h, :] = -(grad_y[:, h+1, :] - grad_y[:, h, :]) / dy
    for w in prange(width - 1):
        img_z[:, :, w] = -(grad_z[:, :, w+1] - grad_z[:, :, w]) / dz

    if edge_strategy == 'Neumann':
        img_x[-1, :, :] = 0
        img_y[:, -1, :] = 0
        img_z[:, :, -1] = 0
    elif edge_strategy == 'Periodic':
        img_x[-1, :, :] = -(grad_x[0, :, :] - grad_x[-1, :, :]) / dx
        img_y[:, -1, :] = -(grad_y[:, 0, :] - grad_y[:, -1, :]) / dy
        img_z[:, :, -1] = -(grad_z[:, :, 0] - grad_z[:, :, -1]) / dz

    image = img_x + img_y + img_z
    return image

@njit(parallel=True)
def central_difference_3d(image, voxel_size=(1,1,1), edge_strategy='replication'):
    dx, dy, dz = voxel_size
    depth, height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    grad_z = np.zeros_like(image)

    for d in prange(1, depth - 1):
        grad_x[d, :, :] = (image[d + 1, :, :] - image[d - 1, :, :]) / (2 * dx)
    for h in prange(1, height - 1):
        grad_y[:, h, :] = (image[:, h + 1, :] - image[:, h - 1, :]) / (2 * dy)
    for w in prange(1, width - 1):
        grad_z[:, :, w] = (image[:, :, w + 1] - image[:, :, w - 1]) / (2 * dz)

    if edge_strategy == 'Neumann':
        grad_x[0, :, :] = grad_x[-1, :, :] = 0
        grad_y[:, 0, :] = grad_y[:, -1, :] = 0
        grad_z[:, :, 0] = grad_z[:, :, -1] = 0
    elif edge_strategy == 'Periodic':
        grad_x[0, :, :] = (image[1, :, :] - image[-1, :, :]) / (2 * dx)
        grad_x[-1, :, :] = (image[0, :, :] - image[-2, :, :]) / (2 * dx)
        grad_y[:, 0, :] = (image[:, 1, :] - image[:, -1, :]) / (2 * dy)
        grad_y[:, -1, :] = (image[:, 0, :] - image[:, -2, :]) / (2 * dy)
        grad_z[:, :, 0] = (image[:, :, 1] - image[:, :, -1]) / (2 * dz)
        grad_z[:, :, -1] = (image[:, :, 0] - image[:, :, -2]) / (2 * dz)

    return grad_x, grad_y, grad_z

@njit(parallel=True)
def central_difference_transpose_3d(grad_x, grad_y, grad_z, voxel_size=(1,1,1), edge_strategy='replication'):
    dx, dy, dz = voxel_size
    depth, height, width = grad_x.shape
    img = np.zeros_like(grad_x)

    for d in prange(1, depth - 1):
        img[d, :, :] = (grad_x[d + 1, :, :] - grad_x[d - 1, :, :]) / (2 * dx)
    for h in prange(1, height - 1):
        img[:, h, :] += (grad_y[:, h + 1, :] - grad_y[:, h - 1, :]) / (2 * dy)
    for w in prange(1, width - 1):
        img[:, :, w] += (grad_z[:, :, w + 1] - grad_z[:, :, w - 1]) / (2 * dz)

    if edge_strategy == 'Neumann':
        img[0, :, :] = img[-1, :, :] = 0
        img[:, 0, :] = img[:, -1, :] = 0
        img[:, :, 0] = img[:, :, -1] = 0
    elif edge_strategy == 'Periodic':
        img[0, :, :] = (grad_x[1, :, :] - grad_x[-1, :, :]) / (2 * dx)
        img[-1, :, :] = (grad_x[0, :, :] - grad_x[-2, :, :]) / (2 * dx)
        img[:, 0, :] += (grad_y[:, 1, :] - grad_y[:, -1, :]) / (2 * dy)
        img[:, -1, :] += (grad_y[:, 0, :] - grad_y[:, -2, :]) / (2 * dy)
        img[:, :, 0] += (grad_z[:, :, 1] - grad_z[:, :, -1]) / (2 * dz)
        img[:, :, -1] += (grad_z[:, :, 0] - grad_z[:, :, -2]) / (2 * dz)

    return img



######################
## Directional Op ####
######################

@njit(parallel=True)
def directional_op(image_gradient, anatomical_gradient, gamma = 1, eta=1e-6):
    """ 
    Calculate the directional operator of a 3D image 
    image_gradient: 3D array of image gradients
    anatomical_gradient: 3D array of anatomical gradients
    """
    out = np.ones_like(image_gradient)
    for d in prange(image_gradient.shape[0]):
        for h in prange(image_gradient.shape[1]):
            for w in prange(image_gradient.shape[2]):
                dot = 0.
                denom = 0.
                for i in range(anatomical_gradient.shape[3]):
                    dot += anatomical_gradient[d,h,w,i] * image_gradient[d,h,w,i]
                    denom += anatomical_gradient[d,h,w,i]**2
                denom += eta**2 # denominator of directional operator - nrm of anatomical gradient squared
                out[d, h, w] = image_gradient[d, h, w] - gamma * anatomical_gradient[d, h, w] * dot / denom # ( 1 - g * xi xi^T ) * image gradient
    return out

@njit(parallel=True)
def directional_op_2d(image_gradient, anatomical_gradient, gamma = 1, eta=1e-6):
    """
    Calculate the directional operator of a 2D image
    image_gradient: 2D array of image gradients
    anatomical_gradient: 2D array of anatomical gradients
    """
    out = np.ones_like(image_gradient)
    for h in prange(image_gradient.shape[0]):
        for w in prange(image_gradient.shape[1]):
            dot = 0.
            denom = 0.
            for i in prange(anatomical_gradient.shape[2]):
                dot += anatomical_gradient[h,w,i] * image_gradient[h,w,i]
                denom += anatomical_gradient[h,w,i]**2
            denom += eta**2
            out[h, w] = image_gradient[h, w] - gamma * anatomical_gradient[h, w] * dot / denom
    return out


### CIL Operators in python ###

import numpy as np

class FiniteDiffCIL(Operator):
    
    def __init__(self, direction = None, method='forward', boundary_condition = 'Neumann', voxel_size=1.0):
        self.method = method
        self.boundary_condition = boundary_condition
        self.voxel_size = voxel_size
        self.size_dom_gm = 3

        if direction is None:
            self.direction=1
        else:
            self.direction = direction

    def get_slice(self, start, stop, end=None):
        
        tmp = [slice(None)]*self.size_dom_gm
        tmp[self.direction] = slice(start, stop, end)
        return tmp  
    
    def direct(self, x, out = None):

        # 
        outa = np.empty_like(x)

        #######################################################################
        ##################### Forward differences #############################
        #######################################################################
                
        if self.method == 'forward':  
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(2, None))], \
                             x[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])               

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(1,2))],\
                            x[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))]) 
                
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(1,2))],\
                            x[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])  
                
            else:
                raise ValueError('Not implemented')                
                
        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                

        elif self.method == 'backward':   
                                   
            # interior nodes
            np.subtract( x[tuple(self.get_slice(1, -1))], \
                             x[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':
                    
                    # right boundary
                    np.subtract( x[tuple(self.get_slice(-1, None))], \
                                 x[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))]) 
                    
            elif self.boundary_condition == 'Periodic':
                  
                # left boundary
                np.subtract(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(-1,None))],\
                            x[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 
        
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################
        
        
        elif self.method == 'centered':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(2, None))], \
                             x[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            
            outa[tuple(self.get_slice(1, -1))] /= 2.
            
            if self.boundary_condition == 'Neumann':
                            
                # left boundary
                np.subtract( x[tuple(self.get_slice(1, 2))], \
                                 x[tuple(self.get_slice(0,1))], \
                                 out = outa[tuple(self.get_slice(0, 1))])  
                outa[tuple(self.get_slice(0, 1))] /=2.
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(-1, None))], \
                                 x[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))])
                outa[tuple(self.get_slice(-1, None))] /=2.                
                
            elif self.boundary_condition == 'Periodic':
                pass
                
               # left boundary
                np.subtract( x[tuple(self.get_slice(1, 2))], \
                                 x[tuple(self.get_slice(-1,None))], \
                                 out = outa[tuple(self.get_slice(0, 1))])                  
                outa[tuple(self.get_slice(0, 1))] /= 2.
                
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(0, 1))], \
                                 x[tuple(self.get_slice(-2,-1))], \
                                 out = outa[tuple(self.get_slice(-1, None))]) 
                outa[tuple(self.get_slice(-1, None))] /= 2.

            else:
                raise ValueError('Not implemented')                 
                
        else:
                raise ValueError('Not implemented')                
        
        if self.voxel_size != 1.0:
            outa /= self.voxel_size  

        return outa                           
                 
        
    def adjoint(self, x, out=None):
        
        # Adjoint operation defined as  

        outa = np.empty_like(x)


            
            
        #######################################################################
        ##################### Forward differences #############################
        #######################################################################            
            

        if self.method == 'forward':    
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(1, -1))], \
                             x[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))])              
            
            if self.boundary_condition == 'Neumann':            

                # left boundary
                outa[tuple(self.get_slice(0,1))] = x[tuple(self.get_slice(0,1))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x[tuple(self.get_slice(-2,-1))]  
                
            elif self.boundary_condition == 'Periodic':            

                # left boundary
                np.subtract(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])  
                # right boundary
                np.subtract(x[tuple(self.get_slice(-1,None))],\
                            x[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 

        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                
                
        elif self.method == 'backward': 
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(2, None))], \
                             x[tuple(self.get_slice(1,-1))], \
                             out = outa[tuple(self.get_slice(1, -1))])             
            
            if self.boundary_condition == 'Neumann':             
                
                # left boundary
                outa[tuple(self.get_slice(0,1))] = x[tuple(self.get_slice(1,2))]                
                
                # right boundary
                outa[tuple(self.get_slice(-1,None))] = - x[tuple(self.get_slice(-1,None))] 
                
                
            elif self.boundary_condition == 'Periodic':
            
                # left boundary
                np.subtract(x[tuple(self.get_slice(1,2))],\
                            x[tuple(self.get_slice(0,1))],
                            out = outa[tuple(self.get_slice(0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(-1,None))])              
                            
            else:
                raise ValueError('Not implemented')
                
                
        #######################################################################
        ##################### Centered differences ############################
        #######################################################################

        elif self.method == 'centered':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(2, None))], \
                             x[tuple(self.get_slice(0,-2))], \
                             out = outa[tuple(self.get_slice(1, -1))]) 
            outa[tuple(self.get_slice(1, -1))] /= 2.0
            

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.add(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(1,2))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0

                # right boundary
                np.add(x[tuple(self.get_slice(-1,None))],\
                            x[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))])  

                outa[tuple(self.get_slice(-1,None))] /= -2.0               
                                                            
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(1,2))],\
                            x[tuple(self.get_slice(-1,None))],
                            out = outa[tuple(self.get_slice(0,1))])
                outa[tuple(self.get_slice(0,1))] /= 2.0
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(0,1))],\
                            x[tuple(self.get_slice(-2,-1))],
                            out = outa[tuple(self.get_slice(-1,None))])
                outa[tuple(self.get_slice(-1,None))] /= 2.0
                
                                
            else:
                raise ValueError('Not implemented') 
                                             
        else:
                raise ValueError('Not implemented')                  
                               
        outa *= -1.
        if self.voxel_size != 1.0:
            outa /= self.voxel_size                      
            
        return outa
    
class GradientCIL(Operator):

    def __init__(self, size_dom_gm=3, method='forward', boundary_condition='Neumann', voxel_size=(1.,1.,1.)):
        self.method = method
        self.boundary_condition = boundary_condition
        self.voxel_size = voxel_size
        self.size_dom_gm = size_dom_gm
        self.operators = []
        
        # Create a finite difference operator for each direction
        for i in range(size_dom_gm):
            self.operators.append(
                FiniteDiffCIL(direction=i, method=method, boundary_condition=boundary_condition, voxel_size=voxel_size[i])
            )

    def direct(self, x, out=None):
        gradient = []
        for op in self.operators:
            gradient.append(op.direct(x))
        return np.stack(gradient, axis=-1)  # The gradient will be added as a new dimension to the output
    
    def adjoint(self, x, out=None):
        adjoint_ops = []
        for i, op in enumerate(self.operators):
            adjoint_ops.append(op.adjoint(x[..., i]))
        return np.sum(adjoint_ops, axis=0)  # Sum up the adjoint operations
    

class JointGradientCIL(Operator):

    def __init__(self, size_dom_gm=3, method='forward', boundary_condition='Neumann', voxel_size=(1.,1.,1.)):
        self.method = method
        self.boundary_condition = boundary_condition
        self.voxel_size = voxel_size
        self.size_dom_gm = size_dom_gm
        self.operators = []
        
        # Create a finite difference operator for each direction
        for i in range(size_dom_gm):
            self.operators.append(
                FiniteDiffCIL(direction=i, method=method, boundary_condition=boundary_condition, voxel_size=voxel_size[i])
            )

    def direct(self, x, out=None):
        # split images by channel
        images = np.split(x, x.shape[-1], axis=-1)
        # reduce singletons in the channel dimension
        images = [np.squeeze(image, axis=-1) for image in images]
        gradient = []
        for image in images:
            for op in self.operators:
                gradient.append(op.direct(image))
        return np.stack(gradient, axis=-1)  # The gradient will be added as a new dimension to the output
    
    def adjoint(self, x, out=None):
        # split images by channel - there will be size_dom_gm images per channel
        images = np.split(x, x.shape[-1] // self.size_dom_gm, axis=-1)
        out = []
        for image in images:
            adjoint_ops = []
            for i, op in enumerate(self.operators):
                adjoint_ops.append(op.adjoint(image[..., i]))
            out.append(np.sum(adjoint_ops, axis=0))  # Sum up the adjoint operations
        return np.stack(out, axis=-1)  # The gradient will be added as a new dimension to the output
