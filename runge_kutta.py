import numpy as np

class RungeKuttaSolverExplicit:

    _name = None

    _tableau_a = []
    _tableau_b = []
    _tableau_c = []

    _n_stages = None

    def __init__(self, problem):
        self._problem = problem

    def step(self, y, t, dt):
        assert self._n_stages != None
        #print(f"Solving at t={t} and dt={dt}")

        k = np.zeros(self._n_stages)
        for i_stage in range(0, self._n_stages):
            y_temp = y
            for j in range(0, i_stage):
                print(f"i={i_stage}, j={j}, a_ij={self._tableau_a[i_stage, j]}")
                print(f"k {k}")
                print(
                    "dt * self._tableau_a[i_stage, j] * k[j]=",
                    dt * self._tableau_a[i_stage, j] * k[j],
                )
                print(f"y_temp={y_temp}")
                y_temp += dt * self._tableau_a[i_stage, j] * k[j]
            print(f"y_temp={y_temp}")
            k[i_stage] = self._problem.evaluate(
                t + self._tableau_c[i_stage] * dt, y_temp
            )

        y_update = 0.0
        for i_stage in range(0, self._n_stages):
            y_update += self._tableau_b[i_stage] * k[i_stage]
        y_new = y + dt * y_update

        return y_new, t + dt

    def report(self):
        print(
            f"{self._name}\nStages:{self._n_stages}\ntableau_a:\n{self._tableau_a}\ntableau_c:\n{self._tableau_c}\ntableau_b:\n{self._tableau_b}\n"
        )
        # print(f"asdf {self._tableau_a}")


class ExplicitEuler(RungeKuttaSolverExplicit):

    _name = "Explicit Euler method"

    _tableau_a = np.array([0])
    _tableau_b = np.array([1])
    _tableau_c = np.array([0])

    _n_stages = 1

    def __init__(self, problem):
        RungeKuttaSolverExplicit.__init__(self, problem)


class ExplicitImprovedEuler(RungeKuttaSolverExplicit):

    _name = "Improved explicit Euler method"

    _tableau_a = np.array([[0.0, 0.0], [0.5, 0.0]])
    _tableau_b = np.array([0.0, 1.0])
    _tableau_c = np.array([0.0, 0.5])

    _n_stages = 2

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

    def __init__(self, problem):
        RungeKuttaSolverExplicit.__init__(self, problem)
