python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 100 
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 100 --device cuda

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 500 
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 500 --device cuda

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 1000
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 1000 --device cuda