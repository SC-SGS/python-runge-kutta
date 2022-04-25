#!/usr/bin/env python3

import rungekuttamethods as rk
import ordinarydifferentialequations
import numpy as np

import unittest


class TestRungeKuttaMethodsSimpleODE(unittest.TestCase):

    _ode = ordinarydifferentialequations.SimpleODE()
    _t0 = 0.0
    _dt = 2**-4
    _t_end = 4.0
    _n_refinements = 4

    def _test_helper(self, runge_kutta_method, expected_errors):
        errors, time_step_sizes = rk.run_convergence_test(
            ode_solver=runge_kutta_method,
            ode=self._ode,
            t0=self._t0,
            dt0=self._dt,
            t_end=self._t_end,
            n_refinements=self._n_refinements,
            expected_order=runge_kutta_method.get_convergence_order(),
        )

        for error, expected_error in zip(errors, expected_errors):
            self.assertAlmostEqual(error, expected_error)

        tolerance = max(0.01, 10 ** (-runge_kutta_method.get_convergence_order()))
        for i in range(2, len(errors)):
            observed_order = np.log(errors[i - 1] / errors[i]) / np.log(
                time_step_sizes[i - 1] / time_step_sizes[i]
            )
            relative_difference = (
                np.linalg.norm(
                    observed_order - runge_kutta_method.get_convergence_order()
                )
                / runge_kutta_method.get_convergence_order()
            )
            self.assertLess(relative_difference, tolerance)

    def test_explicit_euler(self):

        runge_kutta_method = rk.ExplicitEuler()
        expected_errors = [
            1.5100540682874726,
            0.806329526607067,
            0.417029609468097,
            0.21212184860428507,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_improved_explicit_euler(self):

        runge_kutta_method = rk.ExplicitImprovedEuler()
        expected_errors = [
            0.025969110658320194,
            0.006606274724349603,
            0.001665229319667505,
            0.0004179744238743677,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_heun(self):

        runge_kutta_method = rk.Heun()
        expected_errors = [
            0.00030220839979722136,
            3.829837439783823e-05,
            4.819785956122757e-06,
            6.044991813780598e-07,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_classical_runge_kutta(self):

        runge_kutta_method = rk.ClassicalRungeKutta()
        expected_errors = [
            7.085015691687957e-06,
            4.511745324009553e-07,
            2.8462371792770114e-08,
            1.7871712998385192e-09,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_implicit_euler(self):

        runge_kutta_method = rk.ImplicitEuler()
        expected_errors = [
            1.9958629429948083,
            0.9268316393434457,
            0.4470961775885267,
            0.21963481217442204,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_implicit_trapezoidal_rule(self):

        runge_kutta_method = rk.ImplicitTrapezoidalRule()
        expected_errors = [
            0.0002493435738231131,
            6.177528906259333e-05,
            1.5408841388175887e-05,
            3.8500252501627585e-06,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_dirk22(self):

        runge_kutta_method = rk.DIRK22()
        expected_errors = [
            0.006490105775303512,
            0.0016254771671224688,
            0.00040679580219049605,
            0.0001017558749514258,
        ]

        self._test_helper(runge_kutta_method, expected_errors)

    def test_crouzeix_dirk23(self):

        runge_kutta_method = rk.CrouzeixDIRK23()
        expected_errors = [
            0.00129943053795456,
            0.00015671692120533542,
            1.9250463114683924e-05,
            2.385645338165432e-06,
        ]

        self._test_helper(runge_kutta_method, expected_errors)
