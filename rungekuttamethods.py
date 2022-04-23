import numpy as np
import sys


class ButcherTableauException(Exception):
    pass


class RungeKuttaMethod:

    _name = None

    _tableau_a = []
    _tableau_b = []
    _tableau_c = []

    _n_stages = None

    _convergence_order = None

    def _check_tableau(self):
        pass

    def __init__(self, problem):
        self._problem = problem
        try:
            self._check_tableau()
        except ButcherTableauException as err:
            print(err)
            sys.exit(1)

    def step(self, y, t, dt):
        return None, None

    def report(self):
        print(
            f"{self._name}\nStages:{self._n_stages}\ntableau_a:\n{self._tableau_a}\ntableau_c:\n{self._tableau_c}\ntableau_b:\n{self._tableau_b}\n"
        )
        # print(f"asdf {self._tableau_a}")

    def get_problem(self):
        return self._problem

    def get_convergence_order(self):
        return self._convergence_order

    def get_name(self):
        return self._name


class ExplicitRungeKuttaMethod(RungeKuttaMethod):
    def _check_tableau(self):
        if self._tableau_a.shape == (1,):
            if self._tableau_a != 0.0:
                raise ButcherTableauException(
                    f'Butcher tableau of "{self._name}" has wrong non-zero entries making it an implicit method.\n  A[0][0]={self._tableau_a}'
                )
        else:
            if self._tableau_a.shape != (self._n_stages, self._n_stages):
                raise ButcherTableauException(
                    f'Butcher tableau of "{self._name}" has wrong shape. \n  Expected shape ({self._n_stages},{self._n_stages}).\n  Actual shape {self._tableau_a.shape}'
                )
            for i in range(0, self._n_stages):
                for j in range(i, self._n_stages):
                    if self._tableau_a[i][j] != 0.0:
                        raise ButcherTableauException(
                            f'Butcher tableau of "{self._name}" has wrong non-zero entries making it an implicit method.\n  A[{i}][{j}]={self._tableau_a[i][j]}'
                        )

    def step(self, y, t, dt):
        assert self._n_stages != None

        # print(f"{self._n_stages}, {self._problem._system_size}")

        u = np.zeros((self._n_stages, self._problem._system_size, 1))
        for i_stage in range(0, self._n_stages):
            # print( u[i_stage].shape, y.shape, 1 )
            u[i_stage] = y
            for j in range(0, i_stage):
                u[i_stage] = u[i_stage] + (
                    dt
                    * self._tableau_a[i_stage, j]
                    * self._problem.evaluate(t + self._tableau_c[j] * dt, u[j])
                )

        y_new = np.copy(y)
        for i_stage in range(0, self._n_stages):
            y_new = y_new + (
                dt
                * self._tableau_b[i_stage]
                * self._problem.evaluate(t + self._tableau_c[i_stage] * dt, u[i_stage])
            )

        return y_new, t + dt


class ExplicitEuler(ExplicitRungeKuttaMethod):

    _name = "Explicit Euler method"

    _tableau_a = np.array([0])
    _tableau_b = np.array([1])
    _tableau_c = np.array([0])

    _n_stages = 1
    _convergence_order = 1.0

    def __init__(self, problem):
        ExplicitRungeKuttaMethod.__init__(self, problem)


class ExplicitImprovedEuler(ExplicitRungeKuttaMethod):

    _name = "Improved explicit Euler method"

    _tableau_a = np.array([[0.0, 0.0], [0.5, 0.0]])
    _tableau_b = np.array([0.0, 1.0])
    _tableau_c = np.array([0.0, 0.5])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self, problem):
        ExplicitRungeKuttaMethod.__init__(self, problem)


class Heun(ExplicitRungeKuttaMethod):

    _name = "Heun's Runge-Kutta method"

    _tableau_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 0.0],
        ]
    )
    _tableau_b = np.array([1.0 / 4.0, 0, 3.0 / 4.0])
    _tableau_c = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])

    _n_stages = 3
    _convergence_order = 3.0

    def __init__(self, problem):
        ExplicitRungeKuttaMethod.__init__(self, problem)


class ClassicalRungeKutta(ExplicitRungeKuttaMethod):

    _name = "Classical Runge-Kutta method (RK4)"

    _tableau_a = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    _tableau_b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    _tableau_c = np.array([0.0, 0.5, 0.5, 1.0])

    _n_stages = 4
    _convergence_order = 4.0

    def __init__(self, problem):
        ExplicitRungeKuttaMethod.__init__(self, problem)


def solve_ode(ode_integrator, t, dt, t_end, verbose=False, TIME_EPS=1e-12):
    """ """
    # ode_integrator.report()
    time_arr, y_arr = [], []
    y_arr.append(ode_integrator.get_problem().get_initial_value())
    time_arr.append(t)
    y_local = np.copy(ode_integrator.get_problem().get_initial_value())
    while t < t_end - TIME_EPS:
        if verbose:
            print(f"t={t}: Solve for {t+dt}")
        y_local, t = ode_integrator.step(y_local, t, dt)
        y_arr.append(y_local)
        time_arr.append(t)

    return np.array(y_arr), time_arr
