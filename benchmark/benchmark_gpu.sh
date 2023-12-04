python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 100  --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 100 --device cuda --verbose

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 500  --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 500 --device cuda --verbose

python3 benchmark.py --nsteps 1000 --backend jax --ntrials 2 --n 1000 --verbose
python3 benchmark.py --nsteps 1000 --backend torch --ntrials 2 --n 1000 --device cuda --verbose