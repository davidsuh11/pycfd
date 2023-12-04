# PySPH     
PySPH is a basic python implementation of a SPH (Smoothed-Particle Hydrodynamics) based fluid simulation for my final project for MPCS 56430: Scientific Computing. 

## Usage
### CLI 
The most basic usage can done from the through the provided script in the `examples` directory. 
```bash
mkdir output # make output directory 
python3 examples/basic.py
```
This will create a basic video animation (animated through matplotlib) of a 1000-particle system run using SPH, saved to `output/out.mp4`. 

### Python Interface
You can also use the script directly as a module in Python. The exposed module is `simulate`.

```python
import simulate 

# Init simulation
S = simulate.simulation.SPHSimulation(n_particles=1000)

# Step 10000 times
for _ in range(10000):
    S.step() 

S.render_history() # Save to 'output/out.mp4'
```

## Benchmarking
There is also a benchmarking utility Python script provided, which was used in the report. All such utilities are stored in the `benchmark` directory. 
```
usage: benchmark.py [-h] [--n N] [--backend {jax,torch,numpy}] [--ntrials NTRIALS] [--device DEVICE] [--nsteps NSTEPS] [--verbose]

options:
  -h, --help            show this help message and exit
  --n N                 Number of particles
  --backend {jax,torch,numpy}
                        Computational backend to use
  --ntrials NTRIALS     Number of trials to run
  --device DEVICE       Device to run on (only for torch backend)
  --nsteps NSTEPS       Number of steps to run per trial
  --verbose             Verbose results
```
Thus if we want to benchmark a 1000-particle system for 2 trials, 1000 steps each, for pytorch on CUDA, we would run
```bash
cd benchmark
python3 benchmark.py --n 1000 --ntrials 2 --backend torch --device cuda
```
The shell scripts that run all variations of trials for the paper that I used are also included in the `benchmark/`.

## Report
The Project report is available as the report.pdf file. 

## Executive Summary
\* Taken from my paper

Abstract. Executive Summary: SPH (Smoothed-particle Hydrodynamics) is a common method of simulating fluid move-
ment, such as water. It is often more computationally efficient than Euclidean (grid) methods due to nearest-neighbor com-
putations and simplification of the movement equations. The formulas that arise for SPH are ripe for paralleization and
vectorization. Thus, it is of interest if the JAX and pytorch frameworks for python can be utilized to improve simulation per-
formance. We implement a from-scratch version of SPH following the original paper by Mueller from 2003, which can take
numpy, JAX, and pytorch as computational backends and perform benchmarking on a Apple Silicon M2, Nvidia V100 GPU,
and Google TPUv4. Results show that contrary to projections, JAX was not very performant on the GPU (even worse than
its CPU performance), while its TPU performance broken even with CPU. It was the best performer on the CPU, however.
Pytorch on the GPU was extremely performant and showed advantages of framework maturity in the form of computational
optimizations. Numpy as of course the slowest. These results point us to further research into what made JAX more perfor-
mant on the CPU and perhaps implementing these optimizations piecewise into existing systems. It also further opens the
gate for fluid simulations in e.g. game consoles and other edge devices that do not have access to strong GPU hardware.



