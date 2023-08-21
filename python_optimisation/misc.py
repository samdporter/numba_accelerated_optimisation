import numpy as np
from numba import jit, njit, prange

class BlockDataContainer():
    
    def __init__(self, datacontainers: list):
        
        self.containers = np.array(datacontainers)
        
    # function overloading
    def __multiply__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x*y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x*y for y in self.containers])
        
    def __rmultiply__(self, x):
        return self.__multiply__(x)
    
    def __add__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x+y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x+y for y in self.containers])
        
    def __radd__(self, x):
        return self.__add__(x)
    
    def __sub__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x-y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x-y for y in self.containers])
        
    def __rsub__(self, x):
        return self.__sub__(x)
    
    def __truediv__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x/y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x/y for y in self.containers])
        
    def __rtruediv__(self, x):
        return self.__truediv__(x)
    
    def clone(self):
        return BlockDataContainer([x.clone() for x in self.containers])
    
    @property
    def shape(self):
        return self.containers.shape
    

@njit(parallel=True)
def multiply_array(x,y):
    """
    Element-wise multiplication of two 3D arrays.

    Given two 3D arrays `x` and `y`, the function returns a 3D array
    where each element at position (i, j, k) is the product of 
    elements at the same position in `x` and `y`.

    Args:
        x (numpy.ndarray): A 3D array representing the first set of values to be multiplied.
        y (numpy.ndarray): A 3D array representing the second set of values to be multiplied.

    Returns:
        numpy.ndarray: A 3D array containing the result of the element-wise multiplication of `x` and `y`.

    Example:
    If `x` is [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] and `y` is [[[2, 2], [2, 2]], [[2, 2], [2, 2]]],
    the output would be [[[2, 4], [6, 8]], [[10, 12], [14, 16]]].

    Note:
    This function is optimized for parallel execution using Numba's njit decorator.
    """
    out = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                out[i,j,k] = x[i,j,k]*y[i,j,k]
    return out
@njit(parallel=True)


def project_inner(y, lam):
    """
        Perform an element-wise projection of the values in a 4D tensor `y` 
        based on threshold values provided in another 4D tensor `lam`.

        Parameters:
        - y : numpy.ndarray
            A 4D tensor containing the data to be projected.
            Dimensions are assumed to be (i, j, k, l).
            Each element represents a value to be compared against a threshold.
        
        - lam : numpy.ndarray
            A 4D tensor of the same shape as `y`.
            Contains the threshold values for the projection.
            For each corresponding position in `y`, if the absolute value 
            of the element in `y` is less than the threshold in `lam`, 
            the value remains unchanged. Otherwise, it is scaled such that 
            its magnitude equals the threshold while retaining its original sign.

        Returns:
        - numpy.ndarray
            A 4D tensor of the same shape as `y` and `lam`. Contains the projected values.
    """
    out = np.empty_like(y)
    for i in prange(y.shape[0]):
        for j in prange(y.shape[1]):
            for k in prange(y.shape[2]):
                for l in prange(y.shape[3]):
                    val = y[i, j, k, l]
                    abs_val = np.abs(val)
                    if abs_val < lam[i, j, k, l]:
                        out[i, j, k, l] = val
                    else:
                        out[i, j, k, l] = val / abs_val * lam[i, j, k, l]

    return out

@jit(forceobj=True)
def power_iteration(operator, input_shape, num_iterations=100):
    """
    Approximate the largest singular value of an operator using power iteration.

    Args:
        operator: The operator with defined forward and adjoint methods.
        num_iterations (int): The number of iterations to refine the approximation.

    Returns:
        float: The approximated largest singular value of the operator.
    """
    
    # Start with a random input of appropriate shape
    input_data = np.random.randn(*input_shape)
    length = len(input_shape)
    
    for i in range(num_iterations):
        # Apply forward operation
        output_data = operator.forward(input_data)
        
        # Apply adjoint operation
        input_data = operator.adjoint(output_data)
        
        # Normalize the result
        if length == 3:
            norm = fast_norm_parallel_3d(input_data)
        elif length == 4:
            norm = fast_norm_parallel_4d(input_data)

        input_data /= norm

        # print iteration on and remove previous line
        print(f'Iteration {i+1}/{num_iterations}', end='\r')
    
    return norm

@njit(parallel=True)
def fast_norm_parallel_3d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                total += arr[i, j, k]**2

    return np.sqrt(total)

@njit(parallel=True)
def fast_norm_parallel_4d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                for l in prange(s[3]):
                    total += arr[i, j, k, l]**2

    return np.sqrt(total)


@njit
def zero_outside_crop_3d_array(arr, n):
    result = np.zeros_like(arr)
    
    result[n:-n, n:-n, n:-n] = arr[n:-n, n:-n, n:-n]
    
    return result

njit(parallel=True)
def crop_to_cylinder(img, r):
    # Determine the center of the image
    center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
    
    cropped_img = np.zeros_like(img)
    
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            # Calculate distance from the center
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist <= r:
                cropped_img[i, j] = img[i, j]
    
    return cropped_img

