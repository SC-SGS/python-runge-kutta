"""A module to describe ordinary differential equations

This module contains classes to decribe first-order ordinary differential equations such that they can be solved by the Runge-Kutta methods supplied in the :py:mod:`rungekutta` module.
"""
import numpy as np
import sys


class OrdinaryDifferentialEquationException(Exception):
    """Exception to be thrown by ordinary differential equation objects

    An object derived from the :py:class:`OrdinaryDifferentialEquation` class may throw this exception to indicate that:

    1. the initial condition is not set.
    2. the shape of the initial condition is wrong.
    3. private member variables are not set.
    4. a member function is not implemented as expected.
    """
    pass


class OrdinaryDifferentialEquation:
    """Base class to define a first-order ordinary differential equation

    Defines the properties an ordinary differential equation object must have such that it can be solved by the Runge-Kutta methods included in this package. It is assumed that the ordinary differential equations is of first order, i.e., only first order time derivatives appear, and is of shape

    .. math::
        y(t) = f(t, y).

    Additionally, a suitable initial condition

    .. math::
        y(t_0) = y_0

    is known. The ordinary differential equation may be a system of first order differential equations. Higher order differential equations must be rewritten as a system of first-order differential equations.

    The constructor sets the initial condition and checks whether the initial condition was set correctly and has the correct shape. Additionaly it is checked whether the private member variables are set.

    :param initial_condition: Initial condition of the ordinary differential equation. Must be overwritten with sensible initial condition for derived classes., defaults to None
    :type initial_condition: Numpy array, optional
    :raises OrdinaryDifferentialEquationException: The initial condition is not set, incorrect or any of the private member variables is not set.
    """

    _name = None
    _system_size = None

    def _check_initial_condition(self):
        """Checks that initial condition is set and has correct shape.

        :raises OrdinaryDifferentialEquationException: The initial condition is not set or has the wrong shape. Check the error message for details.
        """

        if self._initial_condition is None:
            raise OrdinaryDifferentialEquationException(
                f'Initial condition of "{self._name}" is not set.'
            )

        if self._initial_condition.shape != (self._system_size, 1):
            raise OrdinaryDifferentialEquationException(
                f'Initial condition of "{self._name}" problem has wrong shape. \n  Expected shape ({self._system_size},1).\n  Actual shape {self._initial_condition.shape}'
            )

    def __init__(self, initial_condition=None):
        """Constructor"""
        self._initial_condition = initial_condition

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
        """Evaluate ordinary differential equation at given time and using the given data

        Evaluates the expression:

        .. math::
            y(t) = f(t, y).

        :param t: Time :math:`t`
        :type t: fload
        :param y: Solution :math:`y(t)` at current time :math:`t`
        :type y: Numpy array
        :raises OrdinaryDifferentialEquationException: The function has not been overwritten by the deriving class.
        """
        raise OrdinaryDifferentialEquationException(
            f'Method "evaluate" is not implemented for {self._name}.'
        )

    def evaluate_jacobian(self, t, y):
        """_summary_

        :param t: _description_
        :type t: _type_
        :param y: _description_
        :type y: _type_
        :raises OrdinaryDifferentialEquationException: _description_
        """
        raise OrdinaryDifferentialEquationException(
            f'Method "evaluate_jacobian" is not implemented for {self._name}.'
        )

    def get_initial_condition(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self._initial_condition

    def get_name(self):
        return self._name


class ExponentialFunction(OrdinaryDifferentialEquation):
    """_summary_

    :param OrdinaryDifferentialEquation: _description_
    :type OrdinaryDifferentialEquation: _type_
    :return: _description_
    :rtype: _type_
    """
    _name = "Exponential function"
    _system_size = 1

    def __init__(self, initial_condition=np.array([1.0]), lam=-1.0):
        OrdinaryDifferentialEquation.__init__(self, initial_condition=initial_condition)
        self._lambda = lam

    def evaluate(self, t, y):
        return self._lambda * y

    def evaluate_jacobian(self, t, y):
        return self._lambda

    def exact_solution(self, t):
        return self._initial_condition * np.exp(self._lambda * t)


class SimpleODE(OrdinaryDifferentialEquation):
    """_summary_

    :param OrdinaryDifferentialEquation: _description_
    :type OrdinaryDifferentialEquation: _type_
    :return: _description_
    :rtype: _type_
    """

    _name = "Simple, non-stiff problem y(t)-2*sin(t)"
    _system_size = 1

    def __init__(self, initial_condition=np.array([[1.0]])):
        OrdinaryDifferentialEquation.__init__(self, initial_condition=initial_condition)

    def evaluate(self, t, y):
        return y - 2.0 * np.sin(t)

    def evaluate_jacobian(self, t, y):
        return -2.0 * np.sin(t)

    def exact_solution(self, t):
        return np.sin(t) + np.cos(t)


class VanDerPol(OrdinaryDifferentialEquation):
    """Van der Pol equation class

    The Van der Pol equation is an ordinary differential equation of second order:

    .. math::
        y''(t) = 8*(1-y(t)^2) * y'(t) - y(t).

    The equation is rewritten in this implemenation as a system of ordinary
    differential equations of first order:

    .. math::
        \\begin{equation*}
        y_1'(t) = y_2(t)
        y_2'(t) = 8*(1-y_1(t)^2) * y_2(t) - y_1(t)
        \\end{equation*}.

    The solution to this problem varies strongly in time which
    makes it necessary to use methods of high(er) order and sufficiently small
    time steps. Adaptive time step control is beneficial for this test case.

    :param OrdinaryDifferentialEquation: _description_
    :type OrdinaryDifferentialEquation: _type_
    :return: _description_
    :rtype: _type_
    """

    _name = "van der Pol equation"
    _system_size = 2

    def __init__(self, initial_condition=np.array([[2.0], [0.0]])):
        """_summary_

        :param initial_condition: _description_, defaults to np.array([[2.0], [0.0]])
        :type initial_condition: _type_, optional
        """
        OrdinaryDifferentialEquation.__init__(self, initial_condition=initial_condition)

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
