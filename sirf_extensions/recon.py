from sirf.STIR import TruncateToCylinderProcessor
from sirf_extensions.misc import divide
from python_optimisation.numba_acceleration import *

import matplotlib.pyplot as plt

import os

def OSEM_step_svrg(image, obj_fun, iteration, grads, w, full_grad):
    g = -obj_fun.get_subset_gradient(image, iteration).maximum(0) - grads[iteration] + full_grad
    res = image - w * g
    return res

def OSEM_step(image, obj_fun, iteration, w):
    ratio = obj_fun.get_backprojection_of_acquisition_ratio(image,iteration).maximum(0)
    return w * ratio

def calc_full_grad(image, obj_fun, num_subsets=1):
    full_grad = image.get_uniform_copy(0)
    grads = []
    for i in range(num_subsets):
        grads.append(-obj_fun.get_subset_gradient(image, i).maximum(0))
        full_grad += grads[-1]
    return grads, full_grad/num_subsets

def calc_obj(obj_fun, prior, op, im, beta):
    val = obj_fun(im)
    p = beta * prior(op.direct(im.as_array()))
    return p-val, p , -val
    
def TV_step(im, prior, indic, op, z, w, beta, omega, inner_iters, gamma=1, rho = 0.5, save=True, crop = True):  
    # damping step
    z_bar = omega * z + (1 - omega) * im
    # TV denoising step

    w_arr = beta * w.as_array() # probably not needed

    if crop:
        # crop the image - can be uneven to avoid reducing the size
        crop_size = (im.shape[1])//3
        z_bar_arr = z_bar.as_array()[:, crop_size:-crop_size, crop_size:-crop_size]
        z_arr = z.as_array()[:, crop_size:-crop_size, crop_size:-crop_size]
        w_arr = w_arr[:, crop_size:-crop_size, crop_size:-crop_size]
        
    else:
        z_bar_arr = z_bar.as_array()
        z_arr = z.as_array()
    
    f = L2NormSquared(b = z_arr)
    h = prior * w_arr # only works this way round
    
    sigma = gamma * 2/f.L
    tau = 1 /(gamma * op.norm**2 * sigma)
    pd30 = PD3O()
    pd30.set_up(z_bar_arr, f, indic, h, op, sigma, tau, rho, inner_iters, 1)
    pd30.run(verbose=1)

    if crop:
        # pad solution back to original size
        res = z_bar.as_array()
        res[:, crop_size:-crop_size, crop_size:-crop_size] = pd30.x
    else:
        res = pd30.x
        
    im.fill(res)

    if save:
        return im, pd30.objective

    return im

def run_EMTV(initial, obj_fun, prior, indic, op, beta, epochs=5, num_subsets=8, inner=100, omega=0.5,
                svrg=False, cyl = TruncateToCylinderProcessor(), 
                save_ims=True, path = "", prefix = "", calc_objs=True,
                save_inner_objectives=False, crop=False):
    if calc_objs:
        objectives = []
        priors = []
        datas = []

    im = initial.clone()

    for i in range(epochs):

        # quadratic increase of inner iterations up to inner
        #inner_iters = int(inner * (i+1)/epochs)
        inner_iters = inner

        if svrg and i>=1:
            grads, full_grad = calc_full_grad(im, obj_fun, num_subsets)
            print("Calculated full gradient")
            print(f"len(grads) = {len(grads)}")

        for j in range(num_subsets):
            # OSEM step
            if svrg:
                k = np.random.randint(0, num_subsets)
                print(f"Using subset {k} for SVRG")
            else:
                k=j

            w = divide(im, obj_fun.get_subset_sensitivity(k))+1e-6
            
            if svrg and i>=1:
                z = OSEM_step_svrg(im, obj_fun, k, w, grads, full_grad)
            else:
                z = OSEM_step(im, obj_fun, k, w) 

            cyl.apply(z)

            if save_inner_objectives:
                im, objectives = TV_step(im, prior, indic, op, z, w, beta, omega,  inner_iters, save=True, crop=crop) 
                plt.figure()
                plt.plot(objectives)
                plt.savefig(os.path.join(path,f"inner_objectives_{i}_e_{j}_i_{k}_s.png"))
            else:
                im = TV_step(im, prior, indic, op, z, w, beta, omega, inner_iters, save=False, crop=crop)

            if save_ims:
                im.write(os.path.join(path,prefix+f"_{i}e_{j}i_{k}s.hv"))

            cyl.apply(im)

        if calc_objs:
            obj_val = calc_obj(obj_fun, prior, op, im, beta)

        objectives.append(obj_val[0])
        priors.append(obj_val[1])
        datas.append(obj_val[2])

        print("Objective value: ", obj_val[0])
        print("Prior value: ", obj_val[1])
        print("Data fidelity value: ", obj_val[2])

    return im, objectives, priors, datas
