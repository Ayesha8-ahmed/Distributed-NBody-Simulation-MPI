"""Pygame dashboard. Physics and simulation state live in simulation.py."""

import math
import random

import pygame

from simulation import Simulation, acceleration_for_body


BG = (12, 17, 29)
CARD = (20, 28, 44)
EDGE = (40, 53, 74)
TEXT = (231, 238, 248)
MUTED = (143, 160, 184)
ACCENT = (102, 224, 194)
BLUE = (113, 164, 255)
AMBER = (255, 200, 115)
COUNTS = (5, 10, 50, 100, 500, 1000)
SPEEDS = (0.5, 1.0, 2.0, 4.0)


class Dashboard:
    def __init__(self, screen, state):
        self.screen = screen
        self.state = state
        self.fonts = {size: pygame.font.SysFont("segoeui", size)
                      for size in (12, 13, 14, 15, 16, 18, 22, 28)}
        self.buttons = {}
        self.star_surface = None
        self.layout()

    def layout(self):
        width, height = self.screen.get_size()
        self.canvas = pygame.Rect(24, 112, width - 392, height - 300)
        self.stats = pygame.Rect(24, self.canvas.bottom + 16, self.canvas.w, 74)
        self.future = pygame.Rect(24, self.stats.bottom + 16, self.canvas.w, 58)
        self.controls = pygame.Rect(width - 344, 112, 320, 300)
        self.info = pygame.Rect(width - 344, 428, 320, height - 452)
        self.scale = min((self.canvas.w - 44) / 800, (self.canvas.h - 44) / 600)
        self.origin = (self.canvas.centerx - 400 * self.scale,
                       self.canvas.centery - 300 * self.scale)
        self.buttons = {}
        x, y = self.controls.x + 20, self.controls.y
        self.buttons["run"] = pygame.Rect(x, y + 46, 174, 38)
        self.buttons["reset"] = pygame.Rect(x + 184, y + 46, 96, 38)
        for i, count in enumerate(COUNTS):
            self.buttons[("count", count)] = pygame.Rect(x + (i % 3) * 96,
                                                        y + 119 + (i // 3) * 34, 88, 28)
        for i, speed in enumerate(SPEEDS):
            self.buttons[("speed", speed)] = pygame.Rect(x + i * 72, y + 225, 64, 30)
        self.buttons["trails"] = pygame.Rect(x, y + 266, 280, 24)
        self.buttons["deselect"] = pygame.Rect(self.info.right - 85, self.info.y + 14, 65, 26)
        self.star_surface = None

    def text(self, text, position, size=14, color=TEXT):
        self.screen.blit(self.fonts[size].render(str(text), True, color), position)

    def card(self, rect):
        pygame.draw.rect(self.screen, CARD, rect, border_radius=14)
        pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=14)

    def button(self, key, label, active=False, primary=False):
        rect = self.buttons[key]
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = (37, 52, 73) if hovered else (28, 39, 58)
        if active or primary:
            fill = ACCENT if primary else (35, 70, 74)
        pygame.draw.rect(self.screen, fill, rect, border_radius=7)
        if active:
            pygame.draw.rect(self.screen, ACCENT, rect, 1, border_radius=7)
        surface = self.fonts[14].render(label, True, BG if primary else TEXT)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def world_to_screen(self, x, y):
        return (round(self.origin[0] + x * self.scale),
                round(self.origin[1] + y * self.scale))

    def body_radius(self, mass):
        return max(3, round(max(4, min(13, int(math.sqrt(mass) / 7))) * self.scale))

    def select_body(self, position):
        """Choose the nearest visible body within its click radius."""
        candidates = []
        for i, body in enumerate(self.state.bodies):
            center = self.world_to_screen(body[0], body[1])
            if self.canvas.collidepoint(center):
                distance = math.hypot(center[0] - position[0], center[1] - position[1])
                if distance <= max(10, self.body_radius(body[4]) + 5):
                    candidates.append((distance, i))
        self.state.selected = min(candidates)[1] if candidates else None

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.state.toggle_running()
            elif event.key == pygame.K_r:
                self.state.reset()
            elif event.key == pygame.K_t:
                self.state.toggle_trails()
            elif event.key == pygame.K_ESCAPE:
                self.state.selected = None
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        for key, rect in self.buttons.items():
            if not rect.collidepoint(event.pos):
                continue
            if key == "run":
                self.state.toggle_running()
            elif key == "reset":
                self.state.reset()
            elif key == "trails":
                self.state.toggle_trails()
            elif key == "deselect":
                self.state.selected = None
            elif isinstance(key, tuple) and key[0] == "count":
                self.state.set_count(key[1])
            elif isinstance(key, tuple) and key[0] == "speed":
                self.state.speed = key[1]
            return
        if self.canvas.collidepoint(event.pos):
            self.select_body(event.pos)

    def draw_space(self):
        if self.star_surface is None:
            self.star_surface = pygame.Surface(self.canvas.size)
            self.star_surface.fill((8, 13, 25))
            rng = random.Random(19)
            for _ in range(150):
                pos = (rng.randrange(self.canvas.w), rng.randrange(self.canvas.h))
                pygame.draw.circle(self.star_surface, rng.choice(((36, 48, 68), (58, 72, 94))), pos, 1)
            for x in range(0, self.canvas.w, 80):
                pygame.draw.line(self.star_surface, (15, 23, 38), (x, 0), (x, self.canvas.h))
            for y in range(0, self.canvas.h, 80):
                pygame.draw.line(self.star_surface, (15, 23, 38), (0, y), (self.canvas.w, y))
        self.screen.blit(self.star_surface, self.canvas)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.canvas)
        if self.state.trails_enabled:
            for body, trail in zip(self.state.bodies, self.state.trails):
                if len(trail) > 1:
                    color = tuple(round(c * 0.58) for c in body[5])
                    points = [self.world_to_screen(x, y) for x, y in trail]
                    pygame.draw.lines(self.screen, color, False, points, 1)
        for i, body in enumerate(self.state.bodies):
            center = self.world_to_screen(body[0], body[1])
            radius = self.body_radius(body[4])
            if not self.canvas.inflate(radius * 2, radius * 2).collidepoint(center):
                continue
            pygame.draw.circle(self.screen, body[5], center, radius)
            if i == self.state.selected:
                pygame.draw.circle(self.screen, TEXT, center, radius + 5, 2)
                pygame.draw.circle(self.screen, ACCENT, center, radius + 9, 1)
        self.text("LIVE SIMULATION", (self.canvas.x + 18, self.canvas.y + 14), 12, MUTED)
        self.text("Direct gravity / fixed timestep", (self.canvas.x + 18, self.canvas.bottom - 28), 12, MUTED)
        self.screen.set_clip(previous_clip)
        pygame.draw.rect(self.screen, EDGE, self.canvas, 1, border_radius=10)

    def draw_controls(self):
        self.card(self.controls)
        x, y = self.controls.x + 20, self.controls.y
        self.text("Simulation controls", (x, y + 15), 18)
        label = "Pause" if self.state.running else ("Resume" if self.state.steps else "Start simulation")
        if self.state.max_steps is not None and self.state.steps >= self.state.max_steps:
            label = "Step limit reached"
        self.button("run", label, primary=True)
        self.button("reset", "Reset")
        self.text("BODY COUNT", (x, y + 96), 12, MUTED)
        for count in COUNTS:
            self.button(("count", count), f"{count:,}", len(self.state.bodies) == count)
        self.text("SIMULATION SPEED", (x, y + 201), 12, MUTED)
        for speed in SPEEDS:
            self.button(("speed", speed), f"{speed:g}x", self.state.speed == speed)
        self.button("trails", "Trails: ON" if self.state.trails_enabled else "Trails: OFF",
                    self.state.trails_enabled)

    def draw_inspector(self):
        self.card(self.info)
        x, y = self.info.x + 20, self.info.y
        self.text("Body inspector", (x, y + 16), 18)
        selected = self.state.selected
        if selected is None:
            self.text("Click a body to inspect its properties.", (x, y + 70), 14, MUTED)
            self.text("Select empty space or press Esc to clear.", (x, y + 98), 13, MUTED)
        else:
            self.button("deselect", "Clear")
            body = self.state.bodies[selected]
            ax, ay = acceleration_for_body(self.state.bodies, selected)
            rows = [("Body ID", f"#{selected:04d}"), ("Mass", f"{body[4]:.4f}"),
                    ("Position X", f"{body[0]:.4f}"), ("Position Y", f"{body[1]:.4f}"),
                    ("Velocity X", f"{body[2]:.4f}"), ("Velocity Y", f"{body[3]:.4f}"),
                    ("Speed", f"{math.hypot(body[2], body[3]):.4f}"),
                    ("Acceleration", f"{math.hypot(ax, ay):.4f}")]
            for i, (name, value) in enumerate(rows):
                row_y = y + 58 + i * 27
                self.text(name, (x, row_y), 14, MUTED)
                surface = self.fonts[14].render(value, True, body[5] if i == 0 else TEXT)
                self.screen.blit(surface, (self.info.right - 20 - surface.get_width(), row_y))
        self.text(f"Seed {self.state.seed}  /  dt {self.state.dt:g}",
                  (x, self.info.bottom - 52), 13, MUTED)
        self.text("Simulation units / body IDs start at 0", (x, self.info.bottom - 30), 12, MUTED)

    def draw(self, fps):
        self.screen.fill(BG)
        self.text("N-Body Simulation Lab", (24, 19), 28)
        self.text("Distributed Computing Project", (25, 59), 15, MUTED)
        width = self.screen.get_width()
        status = "Running" if self.state.running else "Paused"
        color = ACCENT if self.state.running else AMBER
        pygame.draw.circle(self.screen, color, (width - 128, 39), 4)
        self.text(status, (width - 114, 26), 18, color)
        self.text("PHASE 1 / SEQUENTIAL", (width - 195, 62), 12, MUTED)
        self.draw_space()
        self.draw_controls()
        self.draw_inspector()
        self.card(self.stats)
        metrics = [("BODIES", f"{len(self.state.bodies):,}"), ("STEPS", f"{self.state.steps:,}"),
                   ("SIMULATED TIME", f"{self.state.simulated_time:.2f} s"),
                   ("FPS", f"{fps:.1f}"), ("MODE", "Sequential")]
        for i, (name, value) in enumerate(metrics):
            x = self.stats.x + 16 + i * (self.stats.w - 24) // 5
            self.text(name, (x, self.stats.y + 12), 12, MUTED)
            self.text(value, (x, self.stats.y + 34), 18, ACCENT if name == "FPS" else TEXT)
        # Deliberately no fabricated MPI or comparison numbers.
        for i, title in enumerate(("MPI workload monitoring", "Sequential vs distributed")):
            rect = pygame.Rect(self.future.x + i * (self.future.w + 12) // 2,
                               self.future.y, (self.future.w - 12) // 2, self.future.h)
            self.card(rect)
            self.text(title, (rect.x + 14, rect.y + 8), 14)
            self.text("Available in later phase", (rect.x + 14, rect.y + 31), 12, MUTED)
        if self.state.running and (self.state.reduced_rate or (0 < fps < 30)):
            self.text("Reduced performance: simulation may run below selected speed",
                      (self.canvas.x + 18, self.canvas.y + 40), 13, AMBER)


def run_dashboard(bodies, steps, dt, seed=42):
    pygame.init()
    try:
        screen = pygame.display.set_mode((1280, 840), pygame.RESIZABLE)
        pygame.display.set_caption("N-Body Simulation Lab")
        state = Simulation(bodies, dt, seed, steps)
        dashboard = Dashboard(screen, state)
        clock = pygame.time.Clock()
        opened = True
        while opened:
            elapsed = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    opened = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(1100, event.w), max(820, event.h)),
                                                     pygame.RESIZABLE)
                    dashboard.screen = screen
                    dashboard.layout()
                else:
                    dashboard.handle_event(event)
            if not opened:
                break
            state.advance(elapsed)
            dashboard.draw(clock.get_fps())
            pygame.display.flip()
            if steps is not None and state.steps >= steps:
                # Keep the completed run visible for inspection and reset.
                state.running = False
    finally:
        pygame.quit()
