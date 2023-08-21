# Module for miscellaneous functions to be used in SIRF
# Should probably be moved to more appropriate location when more functions are added

import numpy as np
import sirf.Reg as reg
from numba import njit, prange
from numpy import expand_dims, random
from sirf.STIR import ImageData
from sirf.Reg import NiftyResample

@njit(parallel=True)
def divide_numba(a,b, eps=1e-6):
    res = np.zeros_like(a)
    tmp = res.ravel()
    for i in prange(a.size):
        if b.flat[i] == 0: # not very technically correct, but it works
            tmp[i] = a.flat[i]/eps
        else:
            tmp[i] = a.flat[i]/b.flat[i]
    return res

def divide(a,b):
    res = a.clone()
    res.fill(divide_numba(a.as_array(), b.as_array()))
    return res

def resample_image(image, reference, interp = 1):
    res = NiftyResample()
    res.set_reference_image(reference)
    res.set_floating_image(image)
    res.set_interpolation_type(interp)
    return res.forward(image)   

def create_image_from_array(array, templ_sino, dims = (1,16,16), voxel_size = (0.4,0.4,0.4)):
    image = ImageData(templ_sino)
    image.initialise(dims, voxel_size)
    # deal with pseudo 3D
    if len(array.shape) == 2:
        array = expand_dims(array, axis = 0)
    image.fill(array)
    return image

def add_noise(proj_data,noise_factor = 0.1, seed = 50):
    """Add Poission noise to acquisition data."""
    proj_data_arr = proj_data.as_array() / noise_factor
    # Data should be >=0 anyway, but add abs just to be safe
    random.seed(seed)
    noisy_proj_data_arr = random.poisson(proj_data_arr).astype('float32');
    noisy_proj_data = proj_data.clone()
    noisy_proj_data.fill(noisy_proj_data_arr*noise_factor);
    return noisy_proj_data

def downsample_max_pool_3D(image, factor):
    """Downsample image by taking the max of a factor x factor x factor box."""
    image_arr = image.as_array()
    image_arr = image_arr.reshape((image_arr.shape[0]//factor, factor, image_arr.shape[1]//factor, factor, image_arr.shape[2]//factor, factor))
    image_arr = image_arr.max(axis=2).max(axis=3).max(axis=4)
    image_arr = image_arr.reshape((image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
    image = image.clone()
    image.fill(image_arr)
    return image

def crop_image_3D(image, new_dims, templ_sino):
    """ 
    Crop Image to new dimensions.
    Even crop is done by taking the center of the image.
    """
    image_arr = image.as_array()
    image_arr = image_arr[(image_arr.shape[0]-new_dims[0])//2:(image_arr.shape[0]+new_dims[0])//2,
                          (image_arr.shape[1]-new_dims[1])//2:(image_arr.shape[1]+new_dims[1])//2,
                          (image_arr.shape[2]-new_dims[2])//2:(image_arr.shape[2]+new_dims[2])//2]
    voxel_size = image.voxel_sizes()
    return create_image_from_array(image_arr, templ_sino, dims = new_dims, voxel_size = voxel_size)

def zoom_image(image, zoom):
    tm_identity = np.array([[1/zoom,0,0,0],
                        [0,1/zoom,0,0],
                        [0,0,1/zoom,0],
                        [0,0,0,1/zoom]])
    TM = reg.AffineTransformation(tm_identity)
    resampler = reg.NiftyResample()
    resampler.set_reference_image(image)
    resampler.set_floating_image(image)
    resampler.add_transformation(TM)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    resampler.process()
    return (resampler.get_output())

def add_noise(proj_data,noise_factor = 0.1, seed = 50):
    """Add Poission noise to acquisition data."""
    proj_data_arr = proj_data.as_array() / noise_factor
    # Data should be >=0 anyway, but add abs just to be safe
    np.random.seed(seed)
    noisy_proj_data_arr = np.random.poisson(proj_data_arr).astype('float32');
    noisy_proj_data = proj_data.clone()
    noisy_proj_data.fill(noisy_proj_data_arr*noise_factor);
    return noisy_proj_data

@njit(parallel=True)
def division(arr1, arr2, num):
    tmp  = np.zeros_like(arr1).flatten()
    for i in prange(tmp.size):
        if arr2.flatten()[i] != 0:
            tmp[i] = arr1.flatten()[i]/arr2.flatten()[i]
        else:
            tmp[i] = arr1.flatten()[i]/num
    return tmp.reshape(arr1.shape)
 