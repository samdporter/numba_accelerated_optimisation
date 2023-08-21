import pytest
import skimage.data as skid
from skimage.util import random_noise
from skimage.transform import resize
import numpy as np

from python_optimisation.operators import Gradient, Jacobian
from sirf_extensions.misc import create_image_from_array

from cil.optimisation.operators import GradientOperator as CILGradientOperator
from sirf.STIR import AcquisitionData

def ignore_boundaries(arr, val=2):
    return arr[val:-val, val:-val, val:-val]

@pytest.fixture(scope='module')
def setup_data():
    template_path = "/home/sam/working/data/template_data/xcat/PET_xcat.hs"
    template = AcquisitionData(template_path)
    arr = skid.kidney()

    arr_resized = resize(arr, (16, 128, 128, 3))

    im_list = [create_image_from_array(arr_resized[:,:,:,i], template, (16, 128, 128)) for i in range(3)]
    grad_cil = CILGradientOperator(im_list[0])
    grad_me = Gradient(voxel_size=im_list[0].voxel_sizes())

    gradient_cil_list = [grad_cil.direct(im) for im in im_list]
    gradient_me_list = [grad_me.direct(arr_resized[:,:,:,i]) for i in range(3)]
    gradient_cil_arrays = [[gradient_cil[i].as_array() for i in range(3)] for gradient_cil in gradient_cil_list]
    gradient_me_arrays = [[gradient_me[:,:,:,2-i] for i in range(3)] for gradient_me in gradient_me_list]
    
    jac_me = Jacobian(voxel_size=im_list[0].voxel_sizes())
    jacobian_me = jac_me.direct(arr_resized)
    jacobian_me_arrays = [[jacobian_me[:,:,:,i,2-j] for j in range(3)] for i in range(3)]

    reverse_gradient_cil_list = [grad_cil.adjoint(gradient_cil) for gradient_cil in gradient_cil_list]
    reverse_gradient_me_list = [grad_me.adjoint(gradient_me) for gradient_me in gradient_me_list]
    reverse_jacobian_me = jac_me.adjoint(jacobian_me)
    
    return gradient_cil_arrays, gradient_me_arrays, jacobian_me_arrays, reverse_gradient_cil_list, reverse_gradient_me_list, reverse_jacobian_me

@pytest.mark.parametrize("index", [0, 1, 2])
def test_gradient_op(setup_data, index):
    gradient_cil_arrays, gradient_me_arrays, _, _, _, _ = setup_data
    for i in range(3):
        np.testing.assert_array_almost_equal(
            ignore_boundaries(gradient_cil_arrays[index][i]),
            ignore_boundaries(gradient_me_arrays[index][i]),
            decimal=4
        )

@pytest.mark.parametrize("index", [0, 1, 2])
def test_jacobian_op(setup_data, index):
    _, gradient_me_arrays, jacobian_me_arrays, _, _, _ = setup_data
    for i in range(3):
        np.testing.assert_array_almost_equal(
            ignore_boundaries(jacobian_me_arrays[index][i]),
            ignore_boundaries(gradient_me_arrays[index][i]),
            decimal=4
        )

@pytest.mark.parametrize("index", [0, 1, 2])
def test_adjoint_gradient_op(setup_data, index):
    _, _, _, reverse_gradient_cil_list, reverse_gradient_me_list, _ = setup_data

    np.testing.assert_array_almost_equal(
        ignore_boundaries(reverse_gradient_cil_list[index].as_array()),
        ignore_boundaries(reverse_gradient_me_list[index]),
        decimal=4
    )

@pytest.mark.parametrize("index", [0, 1, 2])
def test_adjoint_jacobian_op(setup_data, index):
    _, _, _, _, reverse_gradient_me_list, reverse_jacobian_me = setup_data

    np.testing.assert_array_almost_equal(
        ignore_boundaries(reverse_jacobian_me[:,:,:,index]),
        ignore_boundaries(reverse_gradient_me_list[index]),
        decimal=4
    )
