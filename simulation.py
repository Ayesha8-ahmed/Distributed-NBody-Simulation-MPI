"""Display-independent direct N-body physics and dashboard state."""

import math
import random
from collections import deque
from time import perf_counter


WIDTH, HEIGHT = 800, 600
G = 100.0
DT = 0.02
SOFTENING = 10.0


def demo_bodies():
    """Return a fresh copy of the original five-body scene."""
    return [
        [400, 300, 0, 0, 10000, (255, 210, 80)],
        [550, 300, 0, 82, 20, (80, 160, 255)],
        [400, 170, -88, 0, 15, (255, 100, 110)],
        [230, 300, 0, -75, 10, (120, 255, 170)],
        [400, 490, 71, 0, 12, (190, 130, 255)],
    ]


def generate_bodies(count, seed=42):
    """Generate a reproducible scene with one heavy central body."""
    if count < 1:
        raise ValueError("Body count must be positive")
    rng = random.Random(seed)
    bodies = [[WIDTH / 2, HEIGHT / 2, 0.0, 0.0, 10000.0, (255, 210, 80)]]
    for _ in range(count - 1):
        radius = rng.uniform(60.0, 240.0)
        angle = rng.uniform(0.0, math.tau)
        speed = math.sqrt(G * bodies[0][4] * radius**2 /
                          (radius**2 + SOFTENING**2)**1.5)
        bodies.append([
            WIDTH / 2 + radius * math.cos(angle),
            HEIGHT / 2 + radius * math.sin(angle),
            -speed * math.sin(angle), speed * math.cos(angle),
            rng.uniform(1.0, 20.0),
            tuple(rng.randint(80, 255) for _ in range(3)),
        ])
    return bodies


def acceleration_for_body(bodies, index):
    """Acceleration at the current position of one body, in O(N)."""
    body = bodies[index]
    ax, ay = 0.0, 0.0
    for j, other in enumerate(bodies):
        if index == j:
            continue
        dx = other[0] - body[0]
        dy = other[1] - body[1]
        dist_sq = dx * dx + dy * dy + SOFTENING**2
        factor = G * other[4] / (dist_sq * math.sqrt(dist_sq))
        ax += factor * dx
        ay += factor * dy
    return ax, ay


def accelerations(bodies):
    """Calculate all accelerations from the same positions in O(N^2)."""
    return [acceleration_for_body(bodies, i) for i in range(len(bodies))]


def step(bodies, dt=DT):
    """Advance in place using the original semi-implicit Euler method."""
    integrate(bodies, accelerations(bodies), dt)


def integrate(bodies, forces, dt):
    """Commit a complete step after every acceleration has been calculated."""
    for body, (ax, ay) in zip(bodies, forces):
        body[2] += ax * dt
        body[3] += ay * dt
        body[0] += body[2] * dt
        body[1] += body[3] * dt


def cooperative_step(bodies, dt):
    """Yield between force calculations; never expose a partially updated step."""
    forces = []
    for i in range(len(bodies)):
        forces.append(acceleration_for_body(bodies, i))
        yield
    integrate(bodies, forces, dt)


def total_energy(bodies):
    """Kinetic plus potential energy for the softened force law."""
    energy = sum(0.5 * b[4] * (b[2]**2 + b[3]**2) for b in bodies)
    for i, body in enumerate(bodies):
        for other in bodies[i + 1:]:
            distance = math.sqrt((other[0] - body[0])**2 +
                                 (other[1] - body[1])**2 + SOFTENING**2)
            energy -= G * body[4] * other[4] / distance
    return energy


def total_momentum(bodies):
    return (sum(b[4] * b[2] for b in bodies),
            sum(b[4] * b[3] for b in bodies))


def orbit_errors(steps_per_orbit):
    """Measure maximum errors over one softened circular two-body orbit."""
    separation = 100.0
    m1, m2 = 100.0, 50.0
    omega = math.sqrt(G * (m1 + m2) /
                      (separation**2 + SOFTENING**2)**1.5)
    x1 = -separation * m2 / (m1 + m2)
    x2 = separation * m1 / (m1 + m2)
    bodies = [[x1, 0.0, 0.0, omega * x1, m1, (255, 210, 80)],
              [x2, 0.0, 0.0, omega * x2, m2, (80, 160, 255)]]
    initial = [b[:] for b in bodies]
    energy = total_energy(bodies)
    momentum = total_momentum(bodies)
    momentum_scale = sum(b[4] * math.hypot(b[2], b[3]) for b in bodies)
    dt = math.tau / omega / steps_per_orbit
    max_radius_error = max_energy_error = max_momentum_error = 0.0
    for _ in range(steps_per_orbit):
        step(bodies, dt)
        distance = math.hypot(bodies[1][0] - bodies[0][0],
                              bodies[1][1] - bodies[0][1])
        max_radius_error = max(max_radius_error, abs(distance / separation - 1))
        max_energy_error = max(max_energy_error,
                               abs((total_energy(bodies) - energy) / energy))
        current = total_momentum(bodies)
        max_momentum_error = max(max_momentum_error,
                                 math.hypot(current[0] - momentum[0],
                                            current[1] - momentum[1]) / momentum_scale)
    return {
        "separation": max_radius_error,
        "return_position": max(math.hypot(b[0] - original[0], b[1] - original[1]) /
                               separation for b, original in zip(bodies, initial)),
        "energy": max_energy_error,
        "momentum": max_momentum_error,
    }


class Simulation:
    """Fixed-step state with bounded history and cooperative force evaluation."""

    def __init__(self, bodies, dt=DT, seed=42, max_steps=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Timestep must be positive and finite")
        self.dt = dt
        self.seed = seed
        self.max_steps = max_steps
        self.initial = [body[:] for body in bodies]
        self.speed = 1.0
        self.trails_enabled = True
        self.reset()

    def reset(self):
        self.bodies = [body[:] for body in self.initial]
        self.steps = 0
        self.running = False
        self.selected = None
        self.accumulator = 0.0
        self.pending = None
        self.reduced_rate = False
        # Bound both per-body history and total stored points at large N.
        self.trail_length = min(180, max(2, 24000 // len(self.bodies)))
        self.trails = [deque(maxlen=self.trail_length) for _ in self.bodies]

    @property
    def simulated_time(self):
        return self.steps * self.dt

    def set_count(self, count):
        self.initial = generate_bodies(count, self.seed)
        self.reset()

    def toggle_running(self):
        if self.max_steps is not None and self.steps >= self.max_steps:
            return
        self.running = not self.running
        # Discard unfinished calculations (which have not modified bodies).
        self.pending = None
        self.accumulator = 0.0
        self.reduced_rate = False

    def toggle_trails(self):
        self.trails_enabled = not self.trails_enabled
        for trail in self.trails:
            trail.clear()

    def advance(self, elapsed, budget=0.008):
        """Do at most a small wall-time budget; discard excessive time debt.

        The fixed dt never changes with the requested speed or actual FPS.
        Under heavy load simulated time slows instead of skipping physics steps.
        """
        if not self.running:
            return
        elapsed = max(0.0, min(elapsed, 0.1))
        debt = self.accumulator + elapsed * self.speed
        self.accumulator = min(debt, self.dt * 8)
        self.reduced_rate = debt > self.dt * 8 or self.pending is not None
        deadline = perf_counter() + budget
        completed = 0
        while self.accumulator + 1e-12 >= self.dt and completed < 8:
            if perf_counter() >= deadline:
                self.reduced_rate = True
                break
            if self.pending is None:
                self.pending = cooperative_step(self.bodies, self.dt)
            try:
                next(self.pending)
            except StopIteration:
                self.pending = None
                self.accumulator = max(0.0, self.accumulator - self.dt)
                self.steps += 1
                completed += 1
                if self.trails_enabled:
                    for trail, body in zip(self.trails, self.bodies):
                        trail.append((body[0], body[1]))
                if self.max_steps is not None and self.steps >= self.max_steps:
                    self.running = False
                    self.accumulator = 0.0
                    break


