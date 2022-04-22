import numpy as np

class OrdinaryDifferentialEquation:

    _name = None
    _system_size = None

    def __init__(self, initial_value=None):
        self._initial_value=initial_value

    def evaluate(self, t, y):
        return None

    def evaluate_jacobian(self, t, y):
        return None

    def get_initial_value(self):
        return self._initial_value

class ExponentialFunction(OrdinaryDifferentialEquation):

    _name = "Exponential function"
    _system_size = 1

    def __init__(self, initial_value=1.0, lam=-1.0):
        OrdinaryDifferentialEquation.__init__(self, initial_value=initial_value)
        self._lambda = lam

    def evaluate(self, t, y):
        return self._lambda * y

    def evaluate_jacobian(self, t, y):
        return self._lambda

    def exact_solution(self, t ):
        return self._initial_value * np.exp(self._lambda*t)

class SimpleODE(OrdinaryDifferentialEquation):

    _name = "Simple, non-stiff problem y(t)-2*sin(t)"
    _system_size = 1

    def __init__(self, initial_value=1.0):
        OrdinaryDifferentialEquation.__init__(self, initial_value=initial_value)

    def evaluate(self, t, y):
        return y - 2.*np.sin(t)

    def evaluate_jacobian(self, t, y):
        return - 2.*np.sin(t)

    def exact_solution(self, t ):
        return np.sin(t) + np.cos(t)
