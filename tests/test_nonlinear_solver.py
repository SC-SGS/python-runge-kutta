#!/usr/bin/env python3

import nonlinearsolvers as nls
import numpy as np

import unittest


class TestDampedNewtonMethod(unittest.TestCase):
    def test_damped_newton(self):
        x0 = np.array([0, 0])
        f = lambda x: np.array(
            [
                6 * x[0] - np.cos(x[0]) - 2 * x[1],
                8 * x[1] - x[0] * x[1] ** 2 - np.sin(x[0]),
            ]
        )
        df = lambda x: np.array(
            [
                [6.0 + np.sin(x[0]), -2],
                [-x[1] ** 2 - np.cos(x[0]), 8.0 - 2 * x[0] * x[1]],
            ]
        )

        x, n_iterations = nls.damped_newton_raphson(
            f, df, x0, 20, tol_rel=1e-12, tol_abs=1e-12, verbose=False
        )

        self.assertAlmostEqual(x[0], 0.17133364817650473)
        self.assertAlmostEqual(x[1], 0.021321814151378692)
        self.assertEqual(n_iterations, 3)

    def test_damped_newton_single_iteration(self):
        x0 = np.array([0, 0])
        f = lambda x: np.array(
            [
                6 * x[0] - np.cos(x[0]) - 2 * x[1],
                8 * x[1] - x[0] * x[1] ** 2 - np.sin(x[0]),
            ]
        )
        df = lambda x: np.array(
            [
                [6.0 + np.sin(x[0]), -2],
                [-x[1] ** 2 - np.cos(x[0]), 8.0 - 2 * x[0] * x[1]],
            ]
        )

        x, n_iterations = nls.damped_newton_raphson(
            f, df, x0, 1, tol_rel=1e-12, tol_abs=1e-12, verbose=False
        )

        self.assertAlmostEqual(x[0], 8.0 / 46.0)
        self.assertAlmostEqual(x[1], 1.0 / 46.0)
        self.assertEqual(n_iterations, 1)
