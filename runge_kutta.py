import numpy as np


class RungeKuttaSolverExplicit:

    _name = None

    _tableau_a = []
    _tableau_b = []
    _tableau_c = []

    _n_stages = None

    _convergence_order = None

    def __init__(self, problem):
        self._problem = problem

    def step(self, y, t, dt):
        assert self._n_stages != None

        u = np.zeros(self._n_stages)
        for i_stage in range(0, self._n_stages):
            u[i_stage] = y
            for j in range(0, i_stage):
                u[i_stage] += (
                    dt
                    * self._tableau_a[i_stage, j]
                    * self._problem.evaluate(t + self._tableau_c[j] * dt, u[j])
                )

        y_new = y
        for i_stage in range(0, self._n_stages):
            y_new += (
                dt
                * self._tableau_b[i_stage]
                * self._problem.evaluate(t + self._tableau_c[i_stage] * dt, u[i_stage])
            )

        return y_new, t + dt

    #    def step(self, y, t, dt):
    #        assert self._n_stages != None
    #        #print(f"Solving at t={t} and dt={dt}")
    #
    #        k = np.zeros(self._n_stages)
    #        for i_stage in range(0, self._n_stages):
    #            y_temp = y
    #            for j in range(0, i_stage):
    #                print(f"i={i_stage}, j={j}, a_ij={self._tableau_a[i_stage, j]}")
    #                print(f"k {k}")
    #                print(
    #                    "dt * self._tableau_a[i_stage, j] * k[j]=",
    #                    dt * self._tableau_a[i_stage, j] * k[j],
    #                )
    #                print(f"y_temp={y_temp}")
    #                y_temp += dt * self._tableau_a[i_stage, j] * k[j]
    #            print(f"y_temp={y_temp}")
    #            k[i_stage] = self._problem.evaluate(
    #                t + self._tableau_c[i_stage] * dt, y_temp
    #            )
    #
    #        y_update = 0.0
    #        for i_stage in range(0, self._n_stages):
    #            y_update += self._tableau_b[i_stage] * k[i_stage]
    #        y_new = y + dt * y_update
    #
    #        return y_new, t + dt

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


class ExplicitEuler(RungeKuttaSolverExplicit):

    _name = "Explicit Euler method"

    _tableau_a = np.array([0])
    _tableau_b = np.array([1])
    _tableau_c = np.array([0])

    _n_stages = 1
    _convergence_order = 1.0

    def __init__(self, problem):
        RungeKuttaSolverExplicit.__init__(self, problem)


class ExplicitImprovedEuler(RungeKuttaSolverExplicit):

    _name = "Improved explicit Euler method"

    _tableau_a = np.array([[0.0, 0.0], [0.5, 0.0]])
    _tableau_b = np.array([0.0, 1.0])
    _tableau_c = np.array([0.0, 0.5])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self, problem):
        RungeKuttaSolverExplicit.__init__(self, problem)


class Heun(RungeKuttaSolverExplicit):

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
        RungeKuttaSolverExplicit.__init__(self, problem)


class ClassicalRungeKutta(RungeKuttaSolverExplicit):

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
        RungeKuttaSolverExplicit.__init__(self, problem)


def solve_ode( ode_integrator, y, t, dt, t_end, verbose=False, TIME_EPS=1e-12):
    """
    """
    #ode_integrator.report()
    time_arr, y_arr = [], []
    y_arr.append(y[0])
    time_arr.append(t)
    y_local = np.copy(y)
    while t < t_end - TIME_EPS:
        if verbose:
            print(f"t={t}: Solve for {t+dt}")
        y_local, t = ode_integrator.step( y_local, t, dt )
        y_arr.append(y_local)
        time_arr.append(t)

    return y_arr, time_arr