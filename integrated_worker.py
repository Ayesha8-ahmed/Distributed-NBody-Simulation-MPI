"""Sequential and MPI N-body runner with ONE whole-run physics clock.

Protocol (stdout from rank 0 only): ready, started, progress, finished.
Parent sends: RUN <steps>. Initialization happens before the run timer.
The timer measures the entire loop of N physics updates, including inter-rank
communication for MPI, not Python/MPI startup or GUI rendering.
"""
import argparse
import json
import sys
from time import perf_counter


def emit(item):
    print(json.dumps(item, separators=(',', ':'), allow_nan=False), flush=True)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('sequential', 'mpi'), required=True)
    parser.add_argument('--bodies', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dt', type=float, default=.02)
    parser.add_argument('--preview-every', type=int, default=0,
                        help='0 disables preview I/O for clean terminal benchmarks')
    args = parser.parse_args()
    if args.bodies < 1 or args.dt <= 0 or args.preview_every < 0:
        parser.error('bodies and dt must be positive; preview-every must be nonnegative')

    mpi = args.mode == 'mpi'
    if mpi:
        from mpi4py import MPI
        from mpi_engine import MPIEngine
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        engine = MPIEngine(body_count=args.bodies, seed=args.seed, dt=args.dt)
        colors = [b[5] for b in engine.get_bodies()] if rank == 0 else None
        counts = [int(x) for x in engine.counts]
    else:
        from simulation import generate_bodies, step
        rank = 0
        size = 1
        bodies = generate_bodies(args.bodies, args.seed)
        colors = [b[5] for b in bodies]
        counts = [len(bodies)]

    if rank == 0:
        emit({'type': 'ready', 'mode': args.mode, 'bodies': args.bodies,
              'processes': size, 'seed': args.seed, 'dt': args.dt,
              'counts': counts, 'colors': colors})

    command = sys.stdin.readline().strip() if rank == 0 else None
    if mpi:
        command = comm.bcast(command, root=0)
    parts = command.split()
    if len(parts) != 2 or parts[0] != 'RUN':
        raise ValueError('Expected RUN <positive step count>')
    steps = int(parts[1])
    if steps < 1:
        raise ValueError('Step count must be positive')

    # Signal the GUI that initialization is done. MPI ranks synchronize before
    # the timestamp, so ranks measure equivalent boundaries.
    if rank == 0:
        emit({'type': 'started', 'steps_target': steps})
    if mpi:
        comm.Barrier()
    start = perf_counter()
    total_compute = 0.0
    total_comm = 0.0
    last_step_duration = 0.0
    for number in range(1, steps + 1):
        if mpi:
            engine.step(return_bodies=False)
            if rank == 0:
                total_compute += engine.last_computation_time
                total_comm += engine.last_communication_time
        else:
            step_start = perf_counter()
            step(bodies, args.dt)
            last_step_duration = perf_counter() - step_start
            total_compute += last_step_duration
        # Identical preview frequency for both modes. GUI rendering takes
        # place in ANOTHER process; sending snapshots introduces some I/O cost.
        if rank == 0 and args.preview_every and (
                number % args.preview_every == 0 or number == steps):
            positions = (engine.body_state[:, :2].tolist() if mpi
                         else [[b[0], b[1]] for b in bodies])
            emit({'type': 'progress', 'steps': number,
                  'elapsed': perf_counter() - start,
                  'compute': total_compute, 'communication': total_comm,
                  'positions': positions})

    if mpi:
        comm.Barrier()
    local_elapsed = perf_counter() - start
    total_elapsed = (comm.reduce(local_elapsed, op=MPI.MAX, root=0)
                     if mpi else local_elapsed)
    if rank == 0:
        final_positions = (engine.body_state[:, :2].tolist() if mpi
                           else [[b[0], b[1]] for b in bodies])
        emit({'type': 'finished', 'mode': args.mode, 'processes': size,
              'bodies': args.bodies, 'steps': steps, 'seed': args.seed,
              'dt': args.dt, 'elapsed': total_elapsed,
              'compute': total_compute, 'communication': total_comm,
              'counts': counts, 'positions': final_positions})


if __name__ == '__main__':
    run()
