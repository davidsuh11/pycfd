#!/bin/bash 

# This script is used to run the benchmark tests for the project.
python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 100  --verbose
python3 benchmark.py --nsteps 1000 --backend numpy --ntrials 2 --n 100  --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 100  --verbose

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 500  --verbose
python3 benchmark.py --nsteps 1000 --backend numpy --ntrials 2 --n 500  --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 500  --verbose

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 1 --n 1000 --verbose
python3 benchmark.py --nsteps 1000 --backend numpy --ntrials 1 --n 1000 --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 1 --n 1000 --verbose

