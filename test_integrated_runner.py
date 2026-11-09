"""Smoke checks for the fixed-step GUI physics worker. Run: python -m unittest -v test_integrated_runner"""
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent


def worker(mode, n=10, steps=10, ranks=1):
    cmd = [sys.executable, '-u', str(BASE/'integrated_worker.py'),
           '--mode', mode, '--bodies', str(n), '--seed', '42',
           '--dt', '0.02', '--preview-every', '5']
    if mode == 'mpi': cmd = ['mpirun', '--oversubscribe', '-np', str(ranks)] + cmd
    p = subprocess.Popen(cmd, cwd=BASE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        ready_line = p.stdout.readline()
        if not ready_line: raise RuntimeError(p.stderr.read())
        ready = json.loads(ready_line)
        p.stdin.write(f'RUN {steps}\n'); p.stdin.flush()
        result = None
        for line in p.stdout:
            msg = json.loads(line)
            if msg['type'] == 'finished': result = msg; break
        p.wait(timeout=30)
        if p.returncode: raise RuntimeError(p.stderr.read())
        return ready, result
    finally:
        if p.poll() is None: p.kill(); p.wait()
        for stream in (p.stdin, p.stdout, p.stderr):
            if stream is not None: stream.close()


class IntegratedRunnerTests(unittest.TestCase):
    def test_sequential_exact_steps_and_clock(self):
        ready, end = worker('sequential', 10, 10)
        self.assertEqual(ready['type'], 'ready')
        self.assertEqual(end['steps'], 10)
        self.assertEqual(end['bodies'], 10)
        self.assertGreater(end['elapsed'], 0)
        self.assertGreaterEqual(end['elapsed'], end['compute'] * .98)
        self.assertEqual(end['communication'], 0)
        self.assertEqual(len(end['positions']), 10)

    @unittest.skipUnless(shutil.which('mpirun'), 'MPI runtime not installed')
    def test_mpi_exact_steps_and_workload(self):
        ready, end = worker('mpi', 11, 10, 2)
        self.assertEqual(ready['counts'], [6, 5])
        self.assertEqual(end['steps'], 10)
        self.assertEqual(end['processes'], 2)
        self.assertGreater(end['elapsed'], 0)
        self.assertEqual(len(end['positions']), 11)


if __name__ == '__main__':
    unittest.main()
