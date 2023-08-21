# Module for extensions to CIL algorithms 

from cil.framework import BlockDataContainer
import numpy as np

### SPDHG ###

def SPDHG_update_objective(self):
    # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
    p1 = 0.
    for i,op in enumerate(self.operator.operators):
        if i == len(self.operator.operators)-1:
            p2 = self.f[-1](self.operator.operators[-1].direct(self.x)) # save prior term value
            p1 += p2
        else:
            p1 += self.f[i](op.direct(self.x))
    p1 += self.g(self.x)

    p2 = self.f[-1](self.operator.operators[-1].direct(self.x))

    d1 = - self.f.convex_conjugate(self.y_old)
    tmp = self.operator.adjoint(self.y_old)
    tmp *= -1
    d1 -= self.g.convex_conjugate(tmp)

    self.loss.append([p1, d1, p1-d1, p2])

    if self.save_progress is True:
        self.save_images()
    
    if self.update_gamma is True and self.iteration > self.update_iteration:
        self.set_gamma()

def SPDHG_save_images(self):
    if isinstance(self.x, BlockDataContainer):
        for i, x in enumerate(self.x.containers):
            x.write(f"output/x_{self.iteration}_{i}.hv")
            self.z[i].write(f"output/z_{self.iteration}_{i}.hv")
    else:
        self.x.write(f"output/x_{self.iteration}.hv")
        self.z.write(f"output/z_{self.iteration}.hv")


def SPDHG_set_to_initial(self):
    ### Should this set all the variables to the initial values?
    ### i.e x, y, z, zbar, x_tmp
    for i, el in enumerate(self.x):
        if i in self.constant_ims:
            print("Setting {} to initial value \n".format(i))
            el.fill(self.x_init[i])

def SPDHG_set_gamma(self):

    for i in range(len(self.sigma)):
        self.sigma[i] /= self.gamma
        self.tau[i] *= self.gamma

    if isinstance(self.x, BlockDataContainer):
        counter = 0
        for i, x in enumerate(self.x.containers):
            counter += np.sqrt(x.norm())
        self.gamma = counter / len(self.x.containers)
    else:
        self.gamma = np.sqrt(self.x.norm())
    for i in range(len(self.sigma)):
        self.sigma[i] *= self.gamma
        self.tau[i] /= self.gamma

    print(f"\n\nGamma set to {self.gamma}\n\n")

@property
def SPDHG_prior_val(self):
    return [x[3] for x in self.loss]