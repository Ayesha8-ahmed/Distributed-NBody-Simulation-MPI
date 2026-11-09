"""Sequential, softened direct N-body simulation and orbit validation.

Bodies contain [x, y, vx, vy, mass, color]. No graphics dependency is needed
when running headless or checking correctness.
"""

import argparse
import math
import time


from simulation import (DT, G, SOFTENING, WIDTH, HEIGHT, accelerations,
                        demo_bodies, generate_bodies, orbit_errors, step,
                        total_energy, total_momentum)


def check_orbit():
    coarse = orbit_errors(1000)
    fine = orbit_errors(2000)
    limits = {"separation": 0.01, "return_position": 0.01,
              "energy": 0.001, "momentum": 1e-10}
    passed = True
    print("Two-body orbit: relative errors over one orbit")
    print(f"{'Metric':<20} {'1000 steps':>14} {'2000 steps':>14} {'Limit':>14}")
    for name, limit in limits.items():
        ok = all(math.isfinite(errors[name]) and errors[name] <= limit
                 for errors in (coarse, fine))
        passed = passed and ok
        print(f"{name:<20} {coarse[name]:14.6e} {fine[name]:14.6e} {limit:14.6e}"
              f"  {'PASS' if ok else 'FAIL'}")
    converged = all(fine[name] < coarse[name]
                    for name in ("separation", "return_position", "energy"))
    passed = passed and converged
    print(f"Errors decrease when halving the timestep: {'PASS' if converged else 'FAIL'}")
    print(f"Orbit check: {'PASS' if passed else 'FAIL'}")
    return passed


def run_headless(bodies, steps, dt):
    start = time.perf_counter()
    for _ in range(steps):
        step(bodies, dt)
    return time.perf_counter() - start


def run_visual(bodies, steps, dt, seed=42):
    # Keep display dependencies out of headless runs and numerical checks.
    try:
        from gui import run_dashboard
    except ImportError as exc:
        if exc.name != "pygame":
            raise
        raise SystemExit("Visual mode requires pygame: py -m pip install -r requirements.txt") from exc
    run_dashboard(bodies, steps, dt, seed)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", type=positive_int,
                        help="generate N bodies; omit to use the original five-body demo")
    parser.add_argument("--steps", type=positive_int,
                        help="number of steps; default: 1000 headless, unlimited visual")
    parser.add_argument("--seed", type=int, default=42, help="generation seed (default: 42)")
    parser.add_argument("--dt", type=positive_float, default=DT,
                        help="simulation timestep (default: 0.02)")
    parser.add_argument("--headless", action="store_true", help="time simulation without graphics")
    parser.add_argument("--check-orbit", action="store_true",
                        help="run two-body correctness checks and exit")
    args = parser.parse_args(argv)
    if args.check_orbit:
        return 0 if check_orbit() else 1
    bodies = demo_bodies() if args.bodies is None else generate_bodies(args.bodies, args.seed)
    if args.headless:
        steps = args.steps if args.steps is not None else 1000
        elapsed = run_headless(bodies, steps, args.dt)
        print(f"Bodies: {len(bodies)} | Steps: {steps} | dt: {args.dt} | Seed: {args.seed}")
        print(f"Simulation time: {elapsed:.6f} seconds")
        print(f"Mean time per step: {elapsed / steps:.9f} seconds")
        print("Timing includes force calculation and motion updates; excludes setup and graphics.")
    else:
        run_visual(bodies, args.steps, args.dt, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
