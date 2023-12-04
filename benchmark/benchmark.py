import sys
import pathlib
import argparse 
import time 

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import simulate.simulation
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TextColumn

VERBOSE = False
def benchmark_its(n=1000, n_steps = 1000, ntrials = 5, backend='jax', device='cpu', prog=None):
    S = simulate.simulation.SPHSimulation(n_particles=n, backend=backend, device=device)

    # Here we measure the time it takes to compute 1000 steps 
    # of the simulation.
    sm = 0
    task = prog.add_task(f'Benchmarking...', total=ntrials * n_steps)
    for _ in range(ntrials):
        t0 = time.time()
        for _ in range(n_steps):
            S.step()
            prog.update(task, advance=1)

        t1 = time.time()
        sm += t1 - t0
    if VERBOSE:
        print(f"RESULTS: (backend={backend}, device={device}, n={n}, n_steps={n_steps}, ntrials={ntrials})")
        print(f'Average time for {n_steps} steps: {sm / ntrials:.3f}s')
        print(f'Average its/s: {n_steps * ntrials / sm:.3f}')
    else:
        print(f'{sm / ntrials:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--backend', type=str, default='jax')
    parser.add_argument('--ntrials', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    VERBOSE = args.verbose
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), MofNCompleteColumn()) as p:
        benchmark_its(n=args.n, n_steps=args.nsteps, ntrials=args.ntrials, backend=args.backend, device=args.device, prog=p)
