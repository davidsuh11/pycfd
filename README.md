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



