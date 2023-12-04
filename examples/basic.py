import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np 
import sys
import os
import argparse
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import simulate.simulation 


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n-particles', type=int, default=1000)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--p0', type=int, default=1)
    parser.add_argument('--viscosity', type=float, default=0.2)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--steps-per-frame', type=int, default=10)
    parser.add_argument('--border-dims', type=int, nargs=2, default=[20, 80])
    parser.add_argument('--backend', type=str, default='jax')
    parser.add_argument('--h', type=float, default=3)

    args = parser.parse_args()
    simulate.simulation.simulate(n_particles=args.n_particles, 
                        k=args.k, p0=args.p0, 
                        viscosity=args.viscosity, 
                        dt=args.dt, sz=args.border_dims, 
                        num_steps=args.steps,
                        backend=args.backend,)



