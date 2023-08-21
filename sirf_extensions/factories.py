import warnings

from sirf.STIR import (AcquisitionModelUsingMatrix,
                       AcquisitionModelUsingParallelproj,
                       AcquisitionModelUsingRayTracingMatrix,
                       AcquisitionSensitivityModel,
                       PoissonLogLikelihoodWithLinearModelForMeanAndProjData,
                       SeparableGaussianImageFilter, SPECTUBMatrix,
                       OSMAPOSLReconstructor)


### Acquisition Model Factory
def get_acquisition_model(templ_image, templ_sino, detector_efficiency = 1.0, attenuation = True, uMap = None,
                                    norm_sino = None, smooth = False, gauss = (6,5,5), itype = 'PET', num_tangential_LORs = None,
                                    gpu=True):
    """Create Acquisition Model, A, for a system with given attenuation and detector efficiency

    Args:
        uMap (ImageData): Attenuation image 
        templ_sino ([type]): sinogram used as template for projected data 
        detector_efficiency (float, optional): Efficiency of conversion from radiation to signal of detectors. Defaults to 1.0.
        attenuation (bool, optional): Turns attenuation on (True) or off (False). Defaults to True.

    Returns:
        AcquisitonModel: Acquisiton Model of system transforming image space to sinogram space
    """ 
    #%% create acquisition model
    if itype == 'PET':

        if num_tangential_LORs is None:
            num_tangential_LORs = 5

        if gpu is True:
            try:
                am = AcquisitionModelUsingParallelproj()
            except Exception('GPU not available, using CPU instead'):
                am = AcquisitionModelUsingRayTracingMatrix()
                am.set_num_tangential_LORs(num_tangential_LORs)
        else:
            am = AcquisitionModelUsingRayTracingMatrix()
            am.set_num_tangential_LORs(num_tangential_LORs)
        
        if attenuation is True:
            if uMap is None:
                warnings.warn('No attenuation image provided. Continuing without attenuation.')
            else:
                asm_attn = AcquisitionSensitivityModel(uMap, am)
                asm_attn.set_up(templ_sino)
                bin_eff = templ_sino.get_uniform_copy(detector_efficiency)
                bin_eff.fill(detector_efficiency)
                asm_attn.unnormalise(bin_eff)
                asm_attn = AcquisitionSensitivityModel(bin_eff)
                am.set_acquisition_sensitivity(asm_attn)

    elif itype == 'SPECT':
        if gpu is True:
            raise Exception('GPU not available for SPECT')
        acq_model_matrix = SPECTUBMatrix()
        acq_model_matrix.set_resolution_model(0,0,full_3D=True)
        acq_model_matrix.set_keep_all_views_in_cache(False)
        if attenuation is True:
            if uMap is None:
                warnings.warn('No attenuation image provided. Continuing without attenuation.')
            else:
                acq_model_matrix.set_attenuation_image(uMap)
        print(acq_model_matrix.get_keep_all_views_in_cache())
        am = AcquisitionModelUsingMatrix(acq_model_matrix)
    else:
        raise ValueError('Please ensure type = "PET" or "SPECT"')
    
    if smooth is True:
        smoother = SeparableGaussianImageFilter()
        smoother.set_fwhms(gauss)
        am.set_image_data_processor(smoother)

    am.set_up(templ_sino, templ_image)

    return am

def make_pet_acquisition_model(norm_sino, additive, gpu=True):
    if gpu:
        am = AcquisitionModelUsingParallelproj()
    else:
        am = AcquisitionModelUsingRayTracingMatrix()
        am.set_num_tangential_LORs(10)
    asm = AcquisitionSensitivityModel(norm_sino)
    am.set_acquisition_sensitivity(asm)
    am.set_additive_term(additive)
    return am

def make_spect_acquisition_model(umap, cache=False):
    mat = SPECTUBMatrix()
    mat.set_attenuation_image(umap)
    mat.set_keep_all_views_in_cache(cache)
    mat.set_resolution_model(0.03,0.93,full_3D=False)
    am = AcquisitionModelUsingMatrix()
    am.set_matrix(mat)
    return am 

def make_objective_function(sino, am):
    obj_fun = PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    obj_fun.set_acquisition_data(sino)
    obj_fun.set_acquisition_model(am)
    return obj_fun

def make_reconstructor(obj_fun, subsets, subiterations, save_interval, prefix):
    recon = OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(subsets)
    recon.set_num_subiterations(subiterations)
    recon.set_save_interval(save_interval)
    recon.set_output_filename_prefix(prefix)
    return recon