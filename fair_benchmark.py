"""Repeat clean whole-run timing for Sequential and real MPI.

Example:
  python fair_benchmark.py --bodies 100,500 --processes 1,2,4 --steps 100 --repeats 3

Uses the same integrated_worker.py as the GUI, but disables preview snapshots
so performance data is not biased by display/IPC. Startup is excluded.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def measure(mode, bodies, steps, processes=1, seed=42, dt=.02):
    cmd = [sys.executable, '-u', str(BASE / 'integrated_worker.py'),
           '--mode', mode, '--bodies', str(bodies), '--seed', str(seed),
           '--dt', str(dt), '--preview-every', '0']
    if mode == 'mpi':
        cmd = ['mpirun', '--oversubscribe', '-np', str(processes)] + cmd
    process = subprocess.Popen(cmd, cwd=BASE, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=1, text=True)
    try:
        ready_line = process.stdout.readline()
        if not ready_line:
            raise RuntimeError(process.stderr.read()[-1800:] or 'Worker did not initialize')
        ready = json.loads(ready_line)
        if ready.get('type') != 'ready':
            raise RuntimeError('Unexpected worker response: ' + ready_line)
        process.stdin.write(f'RUN {steps}\n')
        process.stdin.flush()
        result = None
        for line in process.stdout:
            message = json.loads(line)
            if message.get('type') == 'finished':
                result = message
                break
        process.wait(timeout=30)
        if process.returncode:
            raise RuntimeError(process.stderr.read()[-1800:])
        if result is None:
            raise RuntimeError('No completed result received')
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bodies', default='100,500')
    parser.add_argument('--processes', default='1,2,4')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    ns = [int(v) for v in args.bodies.split(',')]
    ranks = [int(v) for v in args.processes.split(',')]
    if not all(n > 0 for n in ns + ranks + [args.steps, args.repeats]):
        parser.error('All numeric inputs must be positive')
    rows = []
    for n in ns:
        for mode, p in [('sequential', 1)] + [('mpi', p) for p in ranks]:
            for trial in range(1, args.repeats + 1):
                result = measure(mode, n, args.steps, processes=p)
                row = {k: result[k] for k in ['mode', 'bodies', 'processes',
                        'steps', 'seed', 'dt', 'elapsed', 'compute', 'communication']}
                row['trial'] = trial
                rows.append(row)
                print(f'{mode:10} bodies={n:5} ranks={p} trial={trial} '
                      f'time={result["elapsed"]:.6f}s', flush=True)
    out = BASE / 'benchmarking' / 'fair_results.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ['mode','bodies','processes','steps','seed','dt','trial',
               'elapsed','compute','communication']
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader(); w.writerows(rows)
    print('\nAverages (same N and steps, startup/GUI excluded):')
    for n in ns:
        baseline = statistics.mean(r['elapsed'] for r in rows if
                    r['mode']=='sequential' and r['bodies']==n)
        for mode, p in [('sequential', 1)] + [('mpi', p) for p in ranks]:
            mean = statistics.mean(r['elapsed'] for r in rows if
                   r['mode']==mode and r['processes']==p and r['bodies']==n)
            speedup = baseline / mean if mean else float('nan')
            eff = 100*speedup / p if mode == 'mpi' else 100.
            print(f'N={n:5} {mode:10} P={p:<2} mean={mean:.6f}s '
                  f'speedup={speedup:.3f}x efficiency={eff:.1f}%')
    print('Saved:', out)
    print('Definition: one perf_counter start/end around the entire fixed-step calculation loop.')


if __name__ == '__main__':
    main()
