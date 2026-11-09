"""Numerical and reproducibility checks; run with python -m unittest -v."""

import math
import unittest

import main


class SimulationTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        self.assertEqual(main.generate_bodies(12, 7), main.generate_bodies(12, 7))
        self.assertEqual(len(main.generate_bodies(12, 7)), 12)
        self.assertNotEqual(main.generate_bodies(12, 7), main.generate_bodies(12, 8))

    def test_single_body_moves_at_constant_velocity(self):
        bodies = [[1.0, 2.0, 3.0, -4.0, 10.0, (0, 0, 0)]]
        main.step(bodies, 0.5)
        self.assertEqual(bodies[0][:4], [2.5, 0.0, 3.0, -4.0])

    def test_softened_pair_force_and_simultaneous_update(self):
        bodies = [[0.0, 0.0, 0.0, 0.0, 2.0, (0, 0, 0)],
                  [3.0, 4.0, 0.0, 0.0, 5.0, (0, 0, 0)]]
        factor = main.G / (25 + main.SOFTENING**2)**1.5
        expected = [(factor * 5 * 3, factor * 5 * 4),
                    (-factor * 2 * 3, -factor * 2 * 4)]
        main.step(bodies, 0.1)
        for body, (ax, ay) in zip(bodies, expected):
            self.assertAlmostEqual(body[2], ax * 0.1)
            self.assertAlmostEqual(body[3], ay * 0.1)
        self.assertAlmostEqual(bodies[0][0], expected[0][0] * 0.01)
        self.assertAlmostEqual(bodies[1][0], 3 + expected[1][0] * 0.01)

    def test_coincident_bodies_have_finite_acceleration(self):
        bodies = [[0, 0, 0, 0, 1, (0, 0, 0)], [0, 0, 0, 0, 2, (0, 0, 0)]]
        self.assertEqual(main.accelerations(bodies), [(0.0, 0.0), (0.0, 0.0)])

    def test_orbit_accuracy_and_timestep_convergence(self):
        coarse = main.orbit_errors(1000)
        fine = main.orbit_errors(2000)
        for errors in (coarse, fine):
            for value in errors.values():
                self.assertTrue(math.isfinite(value))
            self.assertLess(errors["separation"], 0.01)
            self.assertLess(errors["return_position"], 0.01)
            self.assertLess(errors["energy"], 0.001)
            self.assertLess(errors["momentum"], 1e-10)
        for name in ("separation", "return_position", "energy"):
            self.assertLess(fine[name], coarse[name])


if __name__ == "__main__":
    unittest.main()
