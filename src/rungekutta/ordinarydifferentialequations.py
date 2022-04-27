import numpy as np
import sys


class OrdinaryDifferentialEquationException(Exception):
    pass


class OrdinaryDifferentialEquation:

    _name = None
    _system_size = None

    def _check_initial_condition(self):
        if self._initial_value.shape != (self._system_size, 1):
            raise OrdinaryDifferentialEquationException(
                f'Initial condition of "{self._name}" problem has wrong shape. \n  Expected shape ({self._system_size},1).\n  Actual shape {self._initial_value.shape}'
            )

    def __init__(self, initial_value=None):
        self._initial_value = initial_value

        if self._name == None:
            raise OrdinaryDifferentialEquationException(
                f'Name property "_name" of class {self.__class__.__name__} is not set.'
            )

        if self._system_size == None:
            raise OrdinaryDifferentialEquationException(
                f'System size property "_system_size" of {self._name} is not set.'
            )

        try:
            self._check_initial_condition()
        except OrdinaryDifferentialEquationException as err:
            print(err)
            sys.exit(1)

    def evaluate(self, t, y):
        raise OrdinaryDifferentialEquationException(
            f'Method "evaluate" is not implemented for {self._name}.'
        )

    def evaluate_jacobian(self, t, y):
        raise OrdinaryDifferentialEquationException(
            f'Method "evaluate_jacobian" is not implemented for {self._name}.'
        )

    def get_initial_value(self):
        return self._initial_value

    def get_name(self):
        return self._name


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
    """Simple"""

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
    """Van der Pol equation class

    The Van der Pol equation is an ordinary differential equation of second order:

        $ y''(t) = 8*(1-y(t)^2) * y'(t) - y(t) $.

    The equation is rewritten in this implemenation as a system of ordinary
    differential equations of first order:

        \begin{equation*}
        y_1'(t) = y_2(t)
        y_2'(t) = 8*(1-y_1(t)^2) * y_2(t) - y_1(t)
        \end{equation*}

    The solution to this problem varies strongly in time which
    makes it necessary to use methods of high(er) order and sufficiently small
    time steps. Adaptive time step control is beneficial for this test case.
    """

    _name = "van der Pol equation"
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
