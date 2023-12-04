import simulate.kernel as kernel
import numpy as np 

import jax.numpy as jnp
import jax

BACKEND = 'jax'

def calculate_pressure_force(pos, p, d, mass, pi, posi, di): 

    multipliers = (-1) * mass * (p + pi) / d * di
    return ( (posi - pos) / 2 * multipliers[:, None]).sum(axis=0) 

__calculate_pressure_force_vmap = jax.jit(jax.vmap(calculate_pressure_force, 
                                                   in_axes=(None, None, None, None, 0, 0, 0)))

def calculate_pressure_force_batch(pos, p, d, mass, dW_spiky_dist):
    fp = __calculate_pressure_force_vmap(pos, p, d, mass, p, pos, dW_spiky_dist)
    fp = np.asarray(fp)
    return fp

def calculate_viscosity_force(pos, v, mass, d, viscosity, vi, di): 
    multipliers = mass / d * viscosity * di
    return ( (v - vi) / 2 * multipliers[:, None] ).sum(axis=0)

__calculate_viscosity_force_vmap = jax.jit(jax.vmap(calculate_viscosity_force,
                                                    in_axes=(None, None, None, None, None, 0, 0)))

def calculate_viscosity_force_batch(pos, v, mass, d, viscosity, lW_viscosity_dist):
    fv = __calculate_viscosity_force_vmap(pos, v, mass, d, viscosity, v, lW_viscosity_dist)
    fv = np.asarray(fv)
    return fv

def __calculate_density(dist_array): 
    return (kernel.jW_poly6(dist_array)).sum(axis=1)

calculate_density_jax = jax.jit(__calculate_density)

def calculate_pressure_force_np(pos, p, d, mass, dW_spiky_dist):
    fp = np.zeros_like(pos)
    for i in range(pos.shape[0]): 
        multipliers = (-1) * mass * (p + p[i]) / d * dW_spiky_dist[i]
        fp[i] += ( (pos[i] - pos) / 2 * multipliers[:, None]).sum(axis=0)  
    return fp

def calculate_viscosity_force_np(pos, v, mass, d, viscosity, lW_viscosity_dist):
    fv = np.zeros_like(pos)
    for i in range(pos.shape[0]): 
        multipliers = mass / d * viscosity \
        * lW_viscosity_dist[i]
        fv[i] += ( (v - v[i]) / 2 * multipliers[:, None] ).sum(axis=0) 
    return fv

def calculate_density_np(dist, mass):
    return kernel.jW_poly6(dist).sum(axis=1)


def calculate_pressure_force(pos, p, d, mass, dW_spiky_dist):
    if BACKEND == 'jax':
        ret = calculate_pressure_force_batch(pos, p, d, mass, dW_spiky_dist)
    elif BACKEND == 'numpy':
        ret = calculate_pressure_force_np(pos, p, d, mass, dW_spiky_dist)
    
    return np.array(ret)

def calculate_viscosity_force(pos, v, mass, d, viscosity, lW_viscosity_dist):
    if BACKEND == 'jax':
        ret = calculate_viscosity_force_batch(pos, v, mass, d, viscosity, lW_viscosity_dist)
    elif BACKEND == 'numpy':
        ret = calculate_viscosity_force_np(pos, v, mass, d, viscosity, lW_viscosity_dist)

    return np.array(ret)

def calculate_density(distarr, mass):
    if BACKEND == 'jax':
        ret = calculate_density_jax(distarr)
    elif BACKEND == 'numpy':
        ret = calculate_density_np(distarr, mass)

    return np.array(ret)
