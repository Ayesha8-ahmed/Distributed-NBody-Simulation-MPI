import sys
import numpy as np

from mpi_engine import MPIEngine
from simulation import generate_bodies, step, DT


def main():
    body_count = int(sys.argv[1]) if len(sys.argv) > 1 else 103
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed = 42

    engine = MPIEngine(
        body_count=body_count,
        seed=seed,
        dt=DT,
    )

    # Run distributed MPI simulation
    for _ in range(steps):
        engine.step()

    passed = None
    if engine.rank == 0:
        mpi_bodies = engine.get_bodies()

        # Run original sequential simulation
        sequential_bodies = generate_bodies(
            body_count,
            seed=seed,
        )

        for _ in range(steps):
            step(sequential_bodies, DT)

        # Compare numerical state:
        # x, y, vx, vy, mass
        mpi_state = np.array([
            body[:5] for body in mpi_bodies
        ])

        sequential_state = np.array([
            body[:5] for body in sequential_bodies
        ])

        difference = np.abs(
            mpi_state - sequential_state
        )

        max_error = np.max(difference)

        print("=" * 60)
        print("SEQUENTIAL vs MPI VERIFICATION")
        print("=" * 60)

        print(f"Bodies         : {body_count}")
        print(f"Steps          : {steps}")
        print(f"MPI processes  : {engine.size}")
        print(f"Seed           : {seed}")
        print(f"DT             : {DT}")
        print(f"Maximum error  : {max_error:.15e}")

        tolerance = 1e-12

        passed = bool(np.allclose(
            mpi_state,
            sequential_state,
            rtol=1e-12,
            atol=tolerance,
        ))
        if passed:
            print("RESULT: PASS")
            print("MPI matches sequential simulation.")
        else:
            print("RESULT: FAIL")
            print("MPI differs from sequential simulation.")

        print("=" * 60)

    passed = engine.comm.bcast(passed, root=0)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

