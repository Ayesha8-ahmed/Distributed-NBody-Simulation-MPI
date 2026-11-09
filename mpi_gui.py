"""
MPI N-Body GUI.

Only MPI rank 0 opens the Pygame window. All ranks participate in every
command broadcast and MPIEngine.step() call.
"""

import math
import random
import sys
import pygame
from mpi4py import MPI

from mpi_engine import MPIEngine
from simulation import DT, generate_bodies, acceleration_for_body

WIDTH, HEIGHT = 1200, 550
BG = (12, 17, 29)
CARD = (20, 28, 44)
EDGE = (40, 53, 74)
TEXT = (231, 238, 248)
MUTED = (143, 160, 184)
ACCENT = (102, 224, 194)
AMBER = (255, 200, 115)
BODY_COUNTS = (10, 103, 500, 1000)


def worker_loop(comm, engine):
    """Non-root ranks wait for root commands and participate in MPI steps."""
    while True:
        command = comm.bcast(None, root=0)

        if command["action"] == "QUIT":
            break
        elif command["action"] == "STEP":
            engine.step()
        elif command["action"] == "RESET":
            engine = MPIEngine(
                body_count=command["body_count"],
                seed=command["seed"],
                dt=command["dt"],
            )


def run_mpi_gui(body_count=103, steps=None, seed=42, dt=DT):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    engine = MPIEngine(body_count=body_count, seed=seed, dt=dt)

    if rank != 0:
        worker_loop(comm, engine)
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("N-Body Simulation Lab — MPI Distributed")
    clock = pygame.time.Clock()
    fonts = {s: pygame.font.SysFont("segoeui", s) for s in (12, 13, 15, 16, 18, 22, 28)}
    rng = random.Random(19)
    stars = [(rng.randrange(WIDTH), rng.randrange(HEIGHT),
              rng.choice(((36, 48, 68), (58, 72, 94)))) for _ in range(180)]

    running = False
    opened = True
    selected = None
    trail_enabled = True
    trails = [[] for _ in range(body_count)]
    accumulator = 0.0
    speed = 1.0
    simulation_steps = 0
    current_count = body_count

    def text(label, pos, size=15, color=TEXT):
        screen.blit(fonts[size].render(str(label), True, color), pos)

    def button(rect, label, active=False):
        color = (35, 70, 74) if active else (28, 39, 58)
        pygame.draw.rect(screen, color, rect, border_radius=7)
        pygame.draw.rect(screen, ACCENT if active else EDGE, rect, 1, border_radius=7)
        surface = fonts[15].render(label, True, TEXT)
        screen.blit(surface, surface.get_rect(center=rect.center))

    def reset_all(new_count):
        nonlocal engine, current_count, simulation_steps, accumulator
        nonlocal selected, trails
        engine = MPIEngine(body_count=new_count, seed=seed, dt=dt)
        current_count = new_count
        simulation_steps = 0
        accumulator = 0.0
        selected = None
        trails = [[] for _ in range(new_count)]

    # Fixed screen-space control locations; layout adapts to window width.
    run_rect = pygame.Rect(30, 105, 150, 42)
    reset_rect = pygame.Rect(190, 105, 110, 42)
    count_rects = {
        count: pygame.Rect(320 + i * 105, 105, 95, 42)
        for i, count in enumerate(BODY_COUNTS)
    }
    trail_rect = pygame.Rect(755, 105, 130, 42)

    while opened:
        elapsed = min(clock.tick(60) / 1000.0, 0.1)
        command = {"action": "IDLE"}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                opened = False
            elif event.type == pygame.VIDEORESIZE:
               screen = pygame.display.set_mode(
                   (event.w, event.h), pygame.RESIZABLE
                 )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                elif event.key == pygame.K_r:
                    command = {
                        "action": "RESET", "body_count": current_count,
                        "seed": seed, "dt": dt
                    }
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    selected = None
                elif event.key == pygame.K_t:
                    trail_enabled = not trail_enabled
                    trails = [[] for _ in range(current_count)]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if run_rect.collidepoint(event.pos):
                    running = not running
                elif reset_rect.collidepoint(event.pos):
                    command = {
                        "action": "RESET", "body_count": current_count,
                        "seed": seed, "dt": dt
                    }
                    running = False
                elif trail_rect.collidepoint(event.pos):
                    trail_enabled = not trail_enabled
                    trails = [[] for _ in range(current_count)]
                else:
                    for count, rect in count_rects.items():
                        if rect.collidepoint(event.pos):
                            command = {
                                "action": "RESET", "body_count": count,
                                "seed": seed, "dt": dt
                            }
                            running = False
                            break

        if not opened:
            comm.bcast({"action": "QUIT"}, root=0)
            break

        if command["action"] == "RESET":
            comm.bcast(command, root=0)
            reset_all(command["body_count"])
        elif running and (steps is None or simulation_steps < steps):
            accumulator += elapsed * speed
            if accumulator >= dt:
                command = {"action": "STEP"}
                accumulator -= dt
                comm.bcast(command, root=0)
                engine.step()
                simulation_steps += 1
                bodies = engine.get_bodies()
                if trail_enabled:
                    for i, body in enumerate(bodies):
                        trails[i].append((body[0], body[1]))
                        if len(trails[i]) > 120:
                            trails[i].pop(0)
            else:
                comm.bcast(command, root=0)
        else:
            running = False
            comm.bcast(command, root=0)

        bodies = engine.get_bodies()

        screen.fill(BG)
        for sx, sy, color in stars:
            pygame.draw.circle(screen, color, (sx % screen.get_width(),
                                               sy % screen.get_height()), 1)

        text("N-Body Simulation Lab", (30, 22), 28)
        text("Distributed Computing — MPI mode", (32, 62), 18, MUTED)
        status = "RUNNING" if running else "PAUSED"
        text(f"{status} | MPI ranks: {size}", (screen.get_width() - 280, 32), 18,
             ACCENT if running else AMBER)

        button(run_rect, "Pause" if running else "Start", active=running)
        button(reset_rect, "Reset")
        for count, rect in count_rects.items():
            button(rect, str(count), active=(current_count == count))
        button(trail_rect, "Trails ON" if trail_enabled else "Trails OFF",
               active=trail_enabled)

        # Main simulation viewport.
        canvas = pygame.Rect(30, 175, screen.get_width() - 360,
                             screen.get_height() - 250)
        pygame.draw.rect(screen, (8, 13, 25), canvas, border_radius=12)
        pygame.draw.rect(screen, EDGE, canvas, 1, border_radius=12)

        scale = min((canvas.w - 30) / 800.0, (canvas.h - 30) / 600.0)
        origin_x = canvas.centerx - 400 * scale
        origin_y = canvas.centery - 300 * scale

        def to_screen(x, y):
            return (round(origin_x + x * scale), round(origin_y + y * scale))

        if trail_enabled:
            for i, trail in enumerate(trails):
                if len(trail) > 1:
                    color = tuple(round(c * 0.55) for c in bodies[i][5])
                    points = [to_screen(x, y) for x, y in trail]
                    pygame.draw.lines(screen, color, False, points, 1)

        for i, body in enumerate(bodies):
            center = to_screen(body[0], body[1])
            radius = max(3, round(max(4, min(13, int(math.sqrt(body[4]) / 7))) * scale))
            if canvas.inflate(radius * 2, radius * 2).collidepoint(center):
                pygame.draw.circle(screen, body[5], center, radius)
                if i == selected:
                    pygame.draw.circle(screen, TEXT, center, radius + 5, 2)

        # Body selection in viewport.
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if canvas.collidepoint(mouse_pos):
                candidates = []
                for i, body in enumerate(bodies):
                    cx, cy = to_screen(body[0], body[1])
                    dist = math.hypot(cx - mouse_pos[0], cy - mouse_pos[1])
                    if dist < 14:
                        candidates.append((dist, i))
                if candidates:
                    selected = min(candidates)[1]
                     # Right-side information panel.
        panel = pygame.Rect(screen.get_width() - 310, 175, 280,
                            screen.get_height() - 250)
        pygame.draw.rect(screen, CARD, panel, border_radius=12)
        pygame.draw.rect(screen, EDGE, panel, 1, border_radius=12)

        px, py = panel.x + 18, panel.y + 18

        text("MPI MONITOR", (px, py), 18)
        text(f"Bodies: {current_count}", (px, py + 42))
        text(f"Completed steps: {simulation_steps}", (px, py + 68))
        text(f"Simulated time: {simulation_steps * dt:.2f}s", (px, py + 94))
        text(f"Rank count: {size}", (px, py + 120))
        text(f"Seed: {seed} | dt: {dt:g}", (px, py + 146), 13, MUTED)

        # Workload distribution across MPI ranks.
        text("Workload Distribution", (px, py + 174), 16)

        workload_rows = (size + 1) // 2

        for row in range(workload_rows):
            y = py + 202 + row * 22
            rank_a = row * 2
            rank_b = rank_a + 1

            if rank_b < size:
                text(
                    f"Rank {rank_a}: {int(engine.counts[rank_a])}  |  "
                    f"Rank {rank_b}: {int(engine.counts[rank_b])}",
                    (px, y), 12
                )
            else:
                text(
                    f"Rank {rank_a}: {int(engine.counts[rank_a])}",
                    (px, y), 12
                )

        # Timing information.
        timing_y = 210 + workload_rows * 22

        text("Last-step timings", (px, py + timing_y), 16)

        text(
            f"Computation: {engine.last_computation_time:.6f}s",
            (px, py + timing_y + 30), 12
        )

        text(
            f"Communication: {engine.last_communication_time:.6f}s",
            (px, py + timing_y + 54), 12
        )

      
        if selected is not None and selected < len(bodies):

            b = bodies[selected]
            text(f"Selected body #{selected}", (px, py + 300), 15, ACCENT)
            text(f"Mass: {b[4]:.3f}", (px, py + 328), 13)
            text(f"X: {b[0]:.3f}", (px, py + 352), 13)
            text(f"Y: {b[1]:.3f}", (px, py + 376), 13)
            text(f"VX: {b[2]:.3f}", (px, py + 400), 13)
            text(f"VY: {b[3]:.3f}", (px, py + 424), 13)

        text("Space: start/pause | R: reset | T: trails | Esc: clear selection",
             (30, screen.get_height() - 38), 13, MUTED)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    body_count = int(sys.argv[1]) if len(sys.argv) > 1 else 103
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_mpi_gui(body_count=body_count, steps=steps)
