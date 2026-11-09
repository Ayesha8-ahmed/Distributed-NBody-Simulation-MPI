"""State and Pygame event checks without opening a desktop window."""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import unittest
import pygame

from gui import Dashboard
from simulation import (DT, Simulation, cooperative_step, demo_bodies,
                        generate_bodies, step)


class StateTests(unittest.TestCase):
    def test_cooperative_step_matches_direct_step_and_is_atomic(self):
        direct = generate_bodies(20)
        cooperative = [body[:] for body in direct]
        pending = cooperative_step(cooperative, DT)
        next(pending)
        self.assertEqual(cooperative, direct)
        list(pending)
        step(direct)
        self.assertEqual(cooperative, direct)

    def test_pause_resume_and_reset_preserve_configuration(self):
        state = Simulation(demo_bodies())
        state.speed = 2.0
        state.toggle_running()
        state.advance(DT, budget=1)
        self.assertEqual(state.steps, 2)
        self.assertTrue(any(state.trails))
        state.toggle_running()
        frozen = [b[:] for b in state.bodies]
        state.advance(1.0)
        self.assertEqual(state.bodies, frozen)
        state.toggle_running()
        state.advance(DT, budget=1)
        self.assertEqual(state.steps, 4)
        state.reset()
        self.assertEqual(state.bodies, demo_bodies())
        self.assertEqual(state.steps, 0)
        self.assertEqual(state.simulated_time, 0)
        self.assertEqual(state.speed, 2.0)
        self.assertFalse(any(state.trails))

    def test_speed_controls_change_step_count_not_timestep(self):
        for speed, expected in ((0.5, 1), (1, 2), (2, 4), (4, 8)):
            state = Simulation(demo_bodies())
            state.speed = speed
            state.toggle_running()
            state.advance(2 * DT, budget=1)
            self.assertEqual(state.steps, expected)
            direct = demo_bodies()
            for _ in range(expected):
                step(direct)
            self.assertEqual(state.bodies, direct)

    def test_pause_and_reset_discard_uncommitted_work(self):
        state = Simulation(demo_bodies())
        state.toggle_running()
        state.pending = cooperative_step(state.bodies, state.dt)
        next(state.pending)
        state.toggle_running()
        self.assertIsNone(state.pending)
        self.assertEqual(state.bodies, demo_bodies())
        state.toggle_running()
        state.pending = cooperative_step(state.bodies, state.dt)
        next(state.pending)
        state.reset()
        self.assertIsNone(state.pending)
        state.toggle_running()
        state.advance(DT, budget=1)
        expected = demo_bodies()
        step(expected)
        self.assertEqual(state.bodies, expected)

    def test_counts_seed_and_safe_reinitialization(self):
        state = Simulation(demo_bodies(), seed=7)
        for count in (5, 10, 50, 100, 500, 1000, 17):
            state.selected = 0
            state.set_count(count)
            self.assertEqual(len(state.bodies), count)
            self.assertEqual(state.bodies, generate_bodies(count, 7))
            self.assertIsNone(state.selected)
            self.assertFalse(state.running)
            self.assertIsNone(state.pending)

    def test_trails_are_bounded_and_can_be_disabled(self):
        state = Simulation(demo_bodies())
        state.toggle_running()
        for _ in range(200):
            state.advance(DT, budget=1)
        self.assertEqual(len(state.trails[0]), state.trail_length)
        self.assertEqual(state.trails[0][-1], tuple(state.bodies[0][:2]))
        state.toggle_trails()
        state.advance(DT, budget=1)
        self.assertFalse(any(state.trails))
        state.reset()
        self.assertFalse(state.trails_enabled)

    def test_large_delays_are_bounded_and_step_limit_is_exact(self):
        state = Simulation(demo_bodies(), max_steps=3)
        state.toggle_running()
        state.advance(100, budget=1)
        self.assertEqual(state.steps, 3)
        self.assertFalse(state.running)
        state.toggle_running()
        self.assertFalse(state.running)
        other = Simulation(demo_bodies())
        other.speed = 4
        other.toggle_running()
        other.advance(100, budget=1)
        self.assertLessEqual(other.steps, 8)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.state = Simulation(demo_bodies())
        self.ui = Dashboard(pygame.display.set_mode((1280, 840)), self.state)

    def click(self, key):
        self.ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                               button=1, pos=self.ui.buttons[key].center))

    def test_control_events_and_inspection(self):
        self.click("run")
        self.state.advance(DT, budget=1)
        self.click("run")
        self.assertFalse(self.state.running)
        self.assertEqual(self.state.steps, 1)
        self.click(("speed", 4.0))
        self.assertEqual(self.state.speed, 4)
        self.click("reset")
        position = self.ui.world_to_screen(*self.state.bodies[1][:2])
        self.ui.select_body(position)
        self.assertEqual(self.state.selected, 1)
        self.ui.draw(60)
        self.click("deselect")
        self.assertIsNone(self.state.selected)
        self.click("trails")
        self.assertFalse(self.state.trails_enabled)
        self.click(("count", 100))
        self.assertEqual(len(self.state.bodies), 100)
        self.ui.draw(60)

    def test_nearest_overlapping_body_and_empty_space(self):
        self.state.bodies[1][:2] = [401, 300]
        self.ui.select_body(self.ui.world_to_screen(401, 300))
        self.assertEqual(self.state.selected, 1)
        self.ui.select_body(self.ui.world_to_screen(10, 10))
        self.assertIsNone(self.state.selected)

    def test_minimum_layout_and_large_scene_render(self):
        self.ui.screen = pygame.display.set_mode((1100, 820))
        self.ui.layout()
        self.state.set_count(1000)
        self.state.selected = 1
        self.ui.draw(30)
        self.assertLessEqual(self.ui.info.bottom, 820)
        self.assertLess(self.ui.canvas.right, self.ui.controls.left)


if __name__ == "__main__":
    unittest.main()
