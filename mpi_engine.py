import sys
from time import perf_counter

import numpy as np
from mpi4py import MPI

from simulation import (
    DT,
    acceleration_for_body,
    generate_bodies,
)


class MPIEngine:
    """
    Distributed N-body engine.

    Each MPI process owns a subset of body indices. Every process receives
    the complete current body state so that acceleration_for_body() uses
    exactly the same positions for the current simulation step.

    Scatterv distributes the assigned body indices.
    Gatherv collects locally updated body states in their original body order.
    """

    def __init__(self, body_count=100, seed=42, dt=DT):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.body_count = body_count
        self.seed = seed
        self.dt = dt

        if body_count < 1:
            raise ValueError("Body count must be positive")

        if self.size not in (1, 2, 4, 8):
            if self.rank == 0:
                print(
                    f"Warning: assignment specifies 1, 2, 4 and 8 processes. "
                    f"Current process count is {self.size}."
                )

        # Rank 0 creates the exact same deterministic initial scene
        # used by Member 1.
        if self.rank == 0:
            self.bodies = generate_bodies(body_count, seed)
        else:
            self.bodies = None

        # Convert body state into a numerical representation that MPI
        # can broadcast efficiently.
        if self.rank == 0:
            self.body_state = self._bodies_to_array(self.bodies)
        else:
            self.body_state = np.empty(
                (body_count, 5),
                dtype=np.float64,
            )

        # Every process gets the complete current body state.
        self.comm.Bcast(self.body_state, root=0)

        # Determine how many bodies each process owns.
        self.counts, self.displacements = self._build_distribution()

        # Scatterv requires the counts/displacements of the body indices.
        self.local_indices = np.empty(
            self.counts[self.rank],
            dtype=np.int32,
        )

        if self.rank == 0:
            all_indices = np.arange(
                body_count,
                dtype=np.int32,
            )
        else:
            all_indices = None

        self.comm.Scatterv(
            [all_indices, self.counts, self.displacements, MPI.INT],
            self.local_indices,
            root=0,
        )

        # Each rank updates and sends only its assigned body states.
        self.local_states = np.empty(
            (self.counts[self.rank], 5), dtype=np.float64
        )
        if self.rank == 0:
            self.all_updated_states = np.empty(
                (body_count, 5), dtype=np.float64
            )
        else:
            self.all_updated_states = None

        self.last_computation_time = 0.0
        self.last_communication_time = 0.0
        self.step_count = 0

    @staticmethod
    def _bodies_to_array(bodies):
        """
        Convert [x, y, vx, vy, mass, color] bodies into numerical data.

        Colors are GUI metadata and are not needed for gravitational
        calculations, so only x, y, vx, vy and mass are distributed.
        """
        result = np.empty(
            (len(bodies), 5),
            dtype=np.float64,
        )

        for i, body in enumerate(bodies):
            result[i, 0] = body[0]
            result[i, 1] = body[1]
            result[i, 2] = body[2]
            result[i, 3] = body[3]
            result[i, 4] = body[4]

        return result

    def _build_distribution(self):
        """
        Evenly distribute body indices, including uneven cases.

        Example:
            103 bodies / 4 processes
            -> 26, 26, 26, 25
        """
        base = self.body_count // self.size
        remainder = self.body_count % self.size

        counts = np.array(
            [
                base + (1 if rank < remainder else 0)
                for rank in range(self.size)
            ],
            dtype=np.int32,
        )

        displacements = np.zeros(
            self.size,
            dtype=np.int32,
        )

        for rank in range(1, self.size):
            displacements[rank] = (
                displacements[rank - 1]
                + counts[rank - 1]
            )

        return counts, displacements

    def _array_to_bodies(self):
        """
        Reconstruct Member 1-compatible body objects.

        Colors are not part of the numerical MPI state, so the original
        colors are recovered from the deterministic initial scene.
        """
        original = generate_bodies(
            self.body_count,
            self.seed,
        )

        bodies = []

        for i in range(self.body_count):
            bodies.append(
                [
                    self.body_state[i, 0],
                    self.body_state[i, 1],
                    self.body_state[i, 2],
                    self.body_state[i, 3],
                    self.body_state[i, 4],
                    original[i][5],
                ]
            )

        return bodies

    def step(self, return_bodies=True):
        """Compute and integrate owned bodies locally, gather, then broadcast."""
        comm_time = 0.0
        current_bodies = self._array_to_bodies()

        comm_start = perf_counter()
        self.comm.Barrier()
        comm_time += perf_counter() - comm_start

        # All accelerations use the OLD positions, before any body moves.
        local_start = perf_counter()
        for local_position, body_index in enumerate(self.local_indices):
            i = int(body_index)
            ax, ay = acceleration_for_body(current_bodies, i)
            x, y, vx, vy, mass = self.body_state[i]

            # Semi-implicit Euler, matching the sequential simulation.
            vx += ax * self.dt
            vy += ay * self.dt
            x += vx * self.dt
            y += vy * self.dt

            self.local_states[local_position] = (x, y, vx, vy, mass)
        local_computation_time = perf_counter() - local_start

        # Collect updated states, in original body order, on rank 0.
        comm_start = perf_counter()
        self.comm.Gatherv(
            self.local_states.ravel(),
            [
                self.all_updated_states.ravel() if self.rank == 0 else None,
                self.counts * 5,
                self.displacements * 5,
                MPI.DOUBLE,
            ],
            root=0,
        )
        comm_time += perf_counter() - comm_start

        if self.rank == 0:
            self.body_state[:] = self.all_updated_states

        # Give every rank the same updated state for the next step.
        comm_start = perf_counter()
        self.comm.Bcast(self.body_state, root=0)
        comm_time += perf_counter() - comm_start

        comm_start = perf_counter()
        self.comm.Barrier()
        comm_time += perf_counter() - comm_start

        self.last_computation_time = self.comm.reduce(
            local_computation_time, op=MPI.MAX, root=0
        )
        self.last_communication_time = self.comm.reduce(
            comm_time, op=MPI.MAX, root=0
        )
        self.step_count += 1
        if self.rank == 0 and return_bodies:
            return self._array_to_bodies()
        return None

    def get_assignment(self):
        """
        Return the actual process assignment.

        Useful for the MPI GUI and workload reporting.
        """
        return {
            "rank": self.rank,
            "processes": self.size,
            "body_count": int(self.counts[self.rank]),
            "indices": self.local_indices.tolist(),
        }

    def get_workload_report(self):
        """
        Gather workload information from all ranks.
        """
        local_report = self.get_assignment()

        reports = self.comm.gather(
            local_report,
            root=0,
        )

        if self.rank == 0:
            return reports

        return None

    def get_bodies(self):
        """Return current Member 1-compatible bodies on rank 0."""
        if self.rank == 0:
            return self._array_to_bodies()

        return None


def run_demo(body_count=103, steps=10, seed=42):
    """
    Small command-line demonstration of the distributed engine.
    """
    engine = MPIEngine(
        body_count=body_count,
        seed=seed,
        dt=DT,
    )

    if engine.rank == 0:
        print("=" * 60)
        print("MPI N-BODY DISTRIBUTED ENGINE")
        print("=" * 60)
        print(f"Bodies       : {body_count}")
        print(f"MPI processes: {engine.size}")
        print(f"Steps        : {steps}")
        print(f"Seed         : {seed}")
        print(f"DT           : {DT}")
        print()

    engine.comm.Barrier()

    for _ in range(steps):
        engine.step()

    reports = engine.get_workload_report()

    if engine.rank == 0:
        print("WORKLOAD DISTRIBUTION")
        print("-" * 60)

        for report in reports:
            print(
                f"Rank {report['rank']}: "
                f"{report['body_count']} bodies "
                f"(indices {report['indices'][0]}.."
                f"{report['indices'][-1]})"
            )

        print()
        print("TIMING")
        print("-" * 60)
        print(
            f"Computation time : "
            f"{engine.last_computation_time:.6f} s"
        )
        print(
            f"Communication time: "
            f"{engine.last_communication_time:.6f} s"
        )
        print("=" * 60)


if __name__ == "__main__":
    body_count = 103
    steps = 10
    seed = 42

    if len(sys.argv) >= 2:
        body_count = int(sys.argv[1])

    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])

    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])

    run_demo(
        body_count=body_count,
        steps=steps,
        seed=seed,
    )
