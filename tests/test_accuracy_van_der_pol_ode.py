#!/usr/bin/env python3

from rungekutta import rungekuttamethods as rk
from rungekutta import ordinarydifferentialequations
import numpy as np

import unittest


class TestExplicitRungeKuttaMethodsVanDerPol(unittest.TestCase):

    _ode = ordinarydifferentialequations.VanDerPol()
    _t0 = 0.0
    _dt = 2**-5
    _t_end = 30.0

    def _test_helper(self, runge_kutta_method, expected_solution):
        print(f"Testing {runge_kutta_method.get_name()} for {self._ode.get_name()}")
        y, _, _ = rk.solve_ode(
            ode_solver=runge_kutta_method,
            ode=self._ode,
            t=self._t0,
            dt=self._dt,
            t_end=self._t_end,
            verbose=False,
        )

        for i, y_i in enumerate(y[-1]):
            self.assertAlmostEqual(y_i.all(), expected_solution[i].all())

    def test_explicit_euler(self):

        runge_kutta_method = rk.ExplicitEuler()
        expected_solution = np.array([[-2.19769405], [0.07161353]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_improved_explicit_euler(self):

        runge_kutta_method = rk.ExplicitImprovedEuler()
        expected_solution = np.array([[-1.25463588], [0.24105237]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_heun(self):

        runge_kutta_method = rk.Heun()
        expected_solution = np.array([[-1.28373875], [0.22431646]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_classical_runge_kutta(self):

        runge_kutta_method = rk.ClassicalRungeKutta()
        expected_solution = np.array([[-1.29636005], [0.21769569]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_implicit_euler(self):

        runge_kutta_method = rk.ImplicitEuler()
        expected_solution = np.array([[1.57263061], [-0.13137364]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_implicit_trapezoidal_rule(self):

        runge_kutta_method = rk.ImplicitTrapezoidalRule()
        expected_solution = np.array([[-1.33044808], [0.20149115]])
        self._test_helper(runge_kutta_method, expected_solution)

    def test_dirk22(self):

        runge_kutta_method = rk.DIRK22()
        expected_solution = np.array([[-1.31220369], [0.20987216]])

        self._test_helper(runge_kutta_method, expected_solution)

    def test_crouzeix_dirk23(self):

        runge_kutta_method = rk.CrouzeixDIRK23()
        expected_solution = np.array([[-1.28231754], [0.22508441]])

        self._test_helper(runge_kutta_method, expected_solution)
