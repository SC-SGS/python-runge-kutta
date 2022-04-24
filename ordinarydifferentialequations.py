import numpy as np
import sys


class OrdinaryDifferentialEquationException(Exception):
    pass


class OrdinaryDifferentialEquation:

    _name = None
    _system_size = None

    def _check_initial_condition(self):

        # if self._system_size == 1:
        #     if self._initial_value.shape != (self._system_size,):
        #         raise OrdinaryDifferentialEquationException(
        #             f'Initial condition of "{self._name}" problem has wrong shape. \n  Expected shape ({self._system_size},).\n  Actual shape {self._initial_value.shape}'
        #         )
        #     return

        if self._initial_value.shape != (self._system_size, 1):
            raise OrdinaryDifferentialEquationException(
                f'Initial condition of "{self._name}" problem has wrong shape. \n  Expected shape ({self._system_size},1).\n  Actual shape {self._initial_value.shape}'
            )

    def __init__(self, initial_value=None):
        self._initial_value = initial_value

        try:
            self._check_initial_condition()
        except OrdinaryDifferentialEquationException as err:
            print(err)
            sys.exit(1)

    def evaluate(self, t, y):
        return None

    def evaluate_jacobian(self, t, y):
        return None

    def get_initial_value(self):
        return self._initial_value


class ExponentialFunction(OrdinaryDifferentialEquation):

    _name = "Exponential function"
    _system_size = 1

    def __init__(self, initial_value=np.array([1.0]), lam=-1.0):
        OrdinaryDifferentialEquation.__init__(self, initial_value=initial_value)
        self._lambda = lam

    def evaluate(self, t, y):
        return self._lambda * y

    def evaluate_jacobian(self, t, y):
        return self._lambda

    def exact_solution(self, t):
        return self._initial_value * np.exp(self._lambda * t)


class SimpleODE(OrdinaryDifferentialEquation):

    _name = "Simple, non-stiff problem y(t)-2*sin(t)"
    _system_size = 1

    def __init__(self, initial_value=np.array([[1.0]])):
        OrdinaryDifferentialEquation.__init__(self, initial_value=initial_value)

    def evaluate(self, t, y):
        return y - 2.0 * np.sin(t)

    def evaluate_jacobian(self, t, y):
        return -2.0 * np.sin(t)

    def exact_solution(self, t):
        return np.sin(t) + np.cos(t)


class VanDerPol(OrdinaryDifferentialEquation):

    _name = "Van der Pol equation"
    _system_size = 2

    def __init__(self, initial_value=np.array([[2.0], [0.0]])):
        OrdinaryDifferentialEquation.__init__(self, initial_value=initial_value)

    def evaluate(self, t, y):

        result = np.zeros((2, 1))
        result[0] = y[1]
        result[1] = 8.0 * (1.0 - y[0] ** 2) * y[1] - y[0]
        return result

    def evaluate_jacobian(self, t, y):
        result = np.zeros((2, 2))

        result[0][0] = 0.0
        result[0][1] = 1.0

        result[1][0] = -16.0 * y[0] * y[1] - 1.0
        result[1][1] = 8.0 * (1.0 - y[0] ** 2)

        return result
