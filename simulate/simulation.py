import numpy as np 
import scipy as sp 

from rich.progress import track 
from rich.live import Live
from rich.table import Table
from rich.console import Console

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os.path import join
import time 

import torch

from . import kernel, calculations

class SPHSimulation:
    def __init__(self, n_particles=100, 
                 dt=0.01, 
                 num_steps=10000, 
                 k=20, p0=1, 
                 viscosity=0.2, 
                 sz=(20, 80),
                 backend='jax',
                 h=3,
                 device='cpu'):
        self.n_particles = n_particles
        self.dt = dt
        self.num_steps = num_steps
        self.k = k
        self.p0 = p0
        self.viscosity = viscosity
        self.sz = sz
        self.mass = np.ones((n_particles,))
        self.hist = []

        self.v = np.zeros((n_particles, 2))
        self.a = np.zeros_like(self.v)
        self.pos = np.random.rand(*self.v.shape) * 15 + kernel.h + 1
        self.output_dir = 'output'
        calculations.BACKEND = backend
        self.backend = backend
        calculations.DEVICE = device
        kernel.h = h

    def step(self):

        d = np.zeros((self.n_particles,)) 
        p = np.zeros((self.n_particles,)) 
        
        fp = np.zeros_like(self.a) 
        fv = np.zeros_like(self.a)
        fe = np.tile([0, -1], (self.n_particles, 1))  

        dist = sp.spatial.distance.cdist(self.pos, self.pos)  

        d = calculations.calculate_density(dist, self.mass)
        # Update the pressure value 
        p = self.k * (d - self.p0) 

        # Update the pressure force value 
        dW_spiky_dist = kernel.dW_spiky(dist)
        fp = calculations.calculate_pressure_force(self.pos, p, d, self.mass, dW_spiky_dist)
        t4 = time.time()

        # Update vis forces
        lW_viscosity_dist = kernel.jlW_viscosity(dist)
        fv = calculations.calculate_viscosity_force(self.pos, self.v, self.mass, d, self.viscosity, lW_viscosity_dist)
        t5 = time.time()

        # Update acceleration, vel, pos 
        forces = fp + fv + fe
        self.a = forces / d[:, None] 
        self.a = np.array(self.a)
        self.v = self.v + self.a * self.dt 
        self.pos = self.pos + self.v * self.dt 
        
        # Boundary conditions 
        sz = self.sz
        xlim = (kernel.h, sz[0] - kernel.h) 
        ylim = (kernel.h, sz[1] - kernel.h) 
        hit_left = self.pos[:, 0] < xlim[0]
        hit_right = self.pos[:, 0] > xlim[1]
        hit_top = self.pos[:, 1] > ylim[1]
        hit_bottom =self.pos[:, 1] < ylim[0]

        self.v = np.array(self.v)
        self.pos = np.array(self.pos)
        self.a = np.array(self.a)
        
        self.v[np.logical_or(hit_left, hit_right), 0] *= -0.6 
        #self.v = calculations.where(np.logical_or(hit_left, hit_right), self.v[:, 0], self.v[:, 0] * -0.6)
        self.pos[hit_left, 0] = xlim[0]  
        #self.pos = calculations.where(hit_left, self.pos[:, 0], xlim[0])
        self.pos[hit_right, 0] = xlim[1] 
        #self.pos = calculations.where(hit_right, self.pos[:, 0], xlim[1])

        self.v[np.logical_or(hit_top, hit_bottom), 1] *= -0.6 
        #calculations.where(np.logical_or(hit_top, hit_bottom), self.v[:, 1], self.v[:, 1] * -0.6)
        self.pos[hit_top, 1] = ylim[1] 
        #calculations.where(hit_top, self.pos[:, 1], ylim[1])
        self.pos[hit_bottom, 1] = ylim[0]
        #calculations.where(hit_bottom, self.pos[:, 1], ylim[0])

        self.hist.append(self.pos) 
    
    def render_history(self, steps_per_frame = 10, out='out.mp4'):
        def animate(i, x=[], y=[], sz=(20, 80)):
            plt.cla() 
            plt.title(f'Simulation with $\\mu$={self.viscosity}, $k$={self.k}, $\\rho_0$={self.p0}' + 
            f', $h$={kernel.h}, $dt$={self.dt}, $n$={self.n_particles}')
            plt.xlim(0, sz[0])
            plt.ylim(0, sz[1])
            cpos = self.hist[i] 

            plt.scatter(cpos[:, 0], cpos[:,1], marker='o')

        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, animate, frames=np.arange(0, len(self.hist), steps_per_frame), interval=50) 
        ani.save(join(self.output_dir, out))


def simulate(n_particles=100, 
             dt=0.01, 
             num_steps=10000, 
             k=20, p0=1, 
             viscosity=0.2, 
             sz=(20, 80),
             backend='jax',):
    sim = SPHSimulation(n_particles=n_particles, dt=dt, num_steps=num_steps, k=k, p0=p0, viscosity=viscosity, sz=sz, backend=backend)
    for t in track(range(num_steps)):
        sim.step()
    
    sim.render_history()
