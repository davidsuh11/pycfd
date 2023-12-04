import numpy as np

from numpy import dot
from numpy.linalg import norm 

import jax 
import jax.numpy as jnp

import torch 
from . import operations

h = 3
d = 2 # 3D 

def W_poly6(r):
    if (type(r) == np.float64 or type(r) == float):
        r = np.array([r]) 
    C = 315 / (64 * np.pi * np.power(h, 9)) 

    ret = C * operations.power(h*h - r*r, 3) 
    ret[r > h] = 0

    return ret

def dW_poly6(r):
    if norm(r) > h: return 0
    C = (-945 / (32 * np.pi * np.power(h,9)))

    return C * np.power(h*h-norm(r), 2) * r


def dW_spiky(r): 
 
    C = (15 / (np.pi * np.power(h, 6)))
    return np.divide(C * (-3) * ((h - r)**2), r, out=np.zeros_like(r),
                     where = np.logical_and(r>1e-15, r < h))

def lW_viscosity(r): 
    ret = (40 / (np.pi * (h**4))) * (1 - r / h) 
    ret[r < 1e-15] = 0 
    ret[r > h] = 0 

    return ret


def __jW_poly6(r):
    C = 315 / (64 * jnp.pi * jnp.power(h, 9)) 
    ret = C * jnp.power(h*h - r*r, 3) 

    return ret

__jW_poly6_jit = jax.jit(__jW_poly6)

def jW_poly6(r):

    ret = __jW_poly6_jit(r)
    # ret = np.array(ret)
    # ret[r > h] = 0
    ret = jnp.where(r > h, 0, ret)
    return ret

def __jlW_viscosity(r): 
    ret = (40 / (np.pi * (h**4))) * (1 - r / h) 
    return ret

__jlW_viscosity_jit = jax.jit(__jlW_viscosity)

def jlW_viscosity(r):
    ret = __jlW_viscosity_jit(r)
    ret = np.array(ret)
    ret[r < 1e-15] = 0 
    ret[r > h] = 0 
    return ret

def tW_poly6(r):
    #r = torch.from_numpy(r)
    C = 315 / (64 * np.pi * np.power(h, 9)) 
    ret = C * torch.pow(h*h - r*r, 3) 
    ret[r > h] = 0

    return ret

# def __jdW_spiky(r): 
#     C = (15 / (np.pi * np.power(h, 6)))
#     return np.divide(C * (-3) * ((h - r)**2), r, out=np.zeros_like(r),
#                      where = np.logical_and(r>1e-15, r < h))

# __jdW_spiky_jit = jax.jit(__jdW_spiky)
# def jdW_spiky(r):
#     ret = __jdW_spiky_jit(r)
#     ret = np.array(ret)
#     return ret