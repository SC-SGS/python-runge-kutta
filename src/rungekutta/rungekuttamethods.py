"""Implementation of explicit and implicit Runge-Kutta methods.

The ordinary differential equations to be solved should be derived from the :py:class:`OrdinaryDifferentialEquation` that is defined in the :py:mod:`ordinarydifferentialequationsmodule`. The module uses the damped Newton method implemented from the :py:mod:`nonlinearsolvers` module if an implicit Runge-Kutta method is used is.
"""

import numpy as np
import sys
from . import nonlinearsolvers


class ButcherTableauException(Exception):
    """Exception to be thrown by Butcher tableaus

    This exception is thrown if:

    1. The Butcher tableau has the wrong shape, i.e. the coefficient vectors for coefficient matrix have the wrong shape.
    2. The Butcher tableau has wrong non-zero pattern which make the method implicit or diagonally implicit if this is not expected.
    """

    pass


class RungeKuttaMethodException(Exception):
    """Exception to be thrown by Runge-Kutta methods

    This exception is thrown if:

    1. If any class member is not set by a class deriving from a Runge-Kutta base class.
    2. If a needed class member function is not overwritten by a class deriving from a Runge-Kutta base class.
    """

    pass


class RungeKuttaMethod:
    """Base class for Runge-Kutta methods

    This class defines the basic members and member functions a Runge-Kutta method should have.

    :raises ButcherTableauException: The Butcher tableau is either not set or does not have th expected shape.
    :raises RungeKuttaMethodException: The mandatory class members or member functions are not overwritten by the deriving class.
    """

    _name = None

    _tableau_a = []
    _tableau_b = []
    _tableau_c = []

    _n_stages = None

    _convergence_order = None

    def _check_tableau(self):
        """Checks whether the Butcher tableau has the expected shape

        :raises ButcherTableauException: The coefficent matrix :math:`A` does not have shape (self._n_stages, self._n_stages)
        :raises ButcherTableauException: The coefficient vector :math:`b` containing the update coefficients does not have shape (self._n_stages,)
        :raises ButcherTableauException: The coefficient vector :math:`c` containing the coefficients for intermediate time steps does not have shape (self._n_stages,)
        """
        if self._tableau_a.shape == (1,) and self._n_stages == 1:
            return

        if self._tableau_a.shape != (self._n_stages, self._n_stages):
            raise ButcherTableauException(
                f'Butcher tableau (a coefficients) of "{self._name}" has wrong shape. \n  Expected shape ({self._n_stages},{self._n_stages}).\n  Actual shape {self._tableau_a.shape}'
            )

        if self._tableau_b.shape != (self._n_stages,):
            raise ButcherTableauException(
                f'Butcher tableau (b coefficients) of "{self._name}" has wrong shape. \n  Expected shape ({self._n_stages},).\n  Actual shape {self._tableau_b.shape}'
            )

        if self._tableau_c.shape != (self._n_stages,):
            raise ButcherTableauException(
                f'Butcher tableau (c coefficients) of "{self._name}" has wrong shape. \n  Expected shape ({self._n_stages},).\n  Actual shape {self._tableau_c.shape}'
            )

    def __init__(self):
        if self._name == None:
            raise RungeKuttaMethodException(
                f'Name property "_name" of class {self.__class__.__name__} is not set.'
            )

        if self._n_stages == None:
            raise RungeKuttaMethodException(
                f'Number of stages property "_n_stages" of {self._name} is not set.'
            )

        if self._n_stages == None:
            raise RungeKuttaMethodException(
                f'Convergence order property "_convergence_order" of {self._name} is not set.'
            )

        try:
            self._check_tableau()
        except ButcherTableauException as err:
            print(err)
            sys.exit(1)

    def step(self, ode, y, t, dt, verbose=False):
        """Compute one time step

        This function computes the solution at a new time step, i.e. it computes

        .. math::
            \\vec{y}^{i+1} = \\vec{y}^{i} + \\sum^{n}_j \\gamma_j \\vec{k}_j

        :param ode: Object describing the ODE.
        :type ode: class OrdinaryDifferentialEquation or any child class.
        :param y: Value to
        :type y: Numpy array
        :param t: Time :math:`t` at starting point.
        :type t: float
        :param dt: Time step size :math:`dt`.
        :type dt: float
        :param verbose: Flag indicating whether the solving procedure should be verbose. If set to `True` intermediate solutions and steps (mainly) of the nonlinear solver will be pritned to screen., defaults to False
        :type verbose: bool, optional
        :raises RungeKuttaMethodException: _description_
        """
        raise RungeKuttaMethodException(
            f'Method "step" is not implemented for {self._name}'
        )

    def report(self):
        """Prints Butcher tableau to screen"""
        print(
            f"{self._name}\nStages:{self._n_stages}\ntableau_a:\n{self._tableau_a}\ntableau_c:\n{self._tableau_c}\ntableau_b:\n{self._tableau_b}\n"
        )

    def get_convergence_order(self):
        """Returns (expected) order of convergence of the method

        :return: Order of convergence
        :rtype: float
        """
        return self._convergence_order

    def get_name(self):
        """Returns name string of Runge-Kutta method

        :return: Name of class
        :rtype: string
        """
        return self._name


class ExplicitRungeKuttaMethod(RungeKuttaMethod):
    def _check_tableau(self):
        RungeKuttaMethod._check_tableau(self)
        if self._tableau_a.shape == (1,):
            if self._tableau_a != 0.0:
                raise ButcherTableauException(
                    f'Butcher tableau of "{self._name}" has wrong non-zero entries making it an implicit method.\n  A[0][0]={self._tableau_a}'
                )
        else:
            for i in range(0, self._n_stages):
                for j in range(i, self._n_stages):
                    if self._tableau_a[i][j] != 0.0:
                        raise ButcherTableauException(
                            f'Butcher tableau of "{self._name}" has wrong non-zero entries making it an implicit method.\n  A[{i}][{j}]={self._tableau_a[i][j]}'
                        )

    def step(self, ode, y, t, dt, verbose=False):
        assert self._n_stages != None

        u = np.zeros((self._n_stages, ode._system_size, 1))
        for i_stage in range(0, self._n_stages):
            u[i_stage] = y
            for j in range(0, i_stage):
                u[i_stage] = u[i_stage] + (
                    dt
                    * self._tableau_a[i_stage, j]
                    * ode.evaluate(t + self._tableau_c[j] * dt, u[j])
                )

        y_new = np.copy(y)
        for i_stage in range(0, self._n_stages):
            y_new = y_new + (
                dt
                * self._tableau_b[i_stage]
                * ode.evaluate(t + self._tableau_c[i_stage] * dt, u[i_stage])
            )

        return y_new, t + dt


class ImplicitRungeKuttaMethod(RungeKuttaMethod):
    def __init__(
        self,
        newton_maximum_iterations=30,
        newton_tolerance_absolute=1e-12,
        newton_tolerance_relative=1e-12,
    ):
        RungeKuttaMethod.__init__(self)
        self._newton_maximum_iterations = newton_maximum_iterations
        self._newton_tolerance_absolute = newton_tolerance_absolute
        self._newton_tolerance_relative = newton_tolerance_relative


class DiagonallyImplicitRungeKuttaMethod(ImplicitRungeKuttaMethod):
    def _check_tableau(self):
        RungeKuttaMethod._check_tableau(self)
        if self._n_stages > 1:
            for i in range(0, self._n_stages):
                for j in range(i + 1, self._n_stages):
                    if self._tableau_a[i][j] != 0.0:
                        raise ButcherTableauException(
                            f'Butcher tableau of "{self._name}" has wrong non-zero entries making it a fully-implicit method.\n  A[{i}][{j}]={self._tableau_a[i][j]}'
                        )

    def step(self, ode, y, t, dt, verbose=False):
        assert self._n_stages != None

        u = np.zeros((self._n_stages, ode._system_size, 1))
        for i_stage in range(0, self._n_stages):
            constant_part = y
            for j in range(0, i_stage):
                constant_part = constant_part + (
                    dt
                    * self._tableau_a[i_stage, j]
                    * ode.evaluate(t + self._tableau_c[j] * dt, u[j])
                )

            # Solve (nonlinear) system of equations
            f = (
                lambda u_i: u_i
                - constant_part
                - dt
                * self._tableau_a[i_stage, i_stage]
                * ode.evaluate(t + self._tableau_c[i_stage] * dt, u_i)
            )

            df = lambda u_i: np.eye(u_i.shape[0]) - dt * self._tableau_a[
                i_stage, i_stage
            ] * ode.evaluate_jacobian(t + self._tableau_c[i_stage] * dt, u_i)

            try:
                u[i_stage], _ = nonlinearsolvers.damped_newton_raphson(
                    f,
                    df,
                    y,
                    max_iter=self._newton_maximum_iterations,
                    tol_rel=self._newton_tolerance_relative,
                    tol_abs=self._newton_tolerance_absolute,
                    verbose=verbose,
                )
            except nonlinearsolvers.NonlinearSolverException as err:
                print(f"Dampled Newton solver did not converge.")
                print(err)
                sys.exit(1)

        y_new = np.copy(y)
        for i_stage in range(0, self._n_stages):
            y_new = y_new + (
                dt
                * self._tableau_b[i_stage]
                * ode.evaluate(t + self._tableau_c[i_stage] * dt, u[i_stage])
            )

        return y_new, t + dt


class ExplicitEuler(ExplicitRungeKuttaMethod):

    _name = "Explicit Euler method"

    _tableau_a = np.array([[0]])
    _tableau_b = np.array([1])
    _tableau_c = np.array([0])

    _n_stages = 1
    _convergence_order = 1.0

    def __init__(self):
        ExplicitRungeKuttaMethod.__init__(self)


class ExplicitImprovedEuler(ExplicitRungeKuttaMethod):

    _name = "Improved explicit Euler method"

    _tableau_a = np.array([[0.0, 0.0], [0.5, 0.0]])
    _tableau_b = np.array([0.0, 1.0])
    _tableau_c = np.array([0.0, 0.5])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self):
        ExplicitRungeKuttaMethod.__init__(self)


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

    def __init__(self):
        ExplicitRungeKuttaMethod.__init__(self)


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

    def __init__(self):
        ExplicitRungeKuttaMethod.__init__(self)


class ImplicitEuler(DiagonallyImplicitRungeKuttaMethod):

    _name = "Implicit Euler method"

    _tableau_a = np.array([[1]])
    _tableau_b = np.array([1])
    _tableau_c = np.array([1])

    _n_stages = 1
    _convergence_order = 1.0

    def __init__(self):
        DiagonallyImplicitRungeKuttaMethod.__init__(self)


class ImplicitTrapezoidalRule(DiagonallyImplicitRungeKuttaMethod):

    _name = "Implicit trapezoidal method"

    _tableau_a = np.array([[0.0, 0.0], [0.5, 0.5]])
    _tableau_b = np.array([0.5, 0.5])
    _tableau_c = np.array([0.0, 1.0])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self):
        DiagonallyImplicitRungeKuttaMethod.__init__(self)


class DIRK22(DiagonallyImplicitRungeKuttaMethod):

    _name = "Two-stage second order DIRK method (DIRK22)"

    _coefficient = (
        1.0 - np.sqrt(2) / 2.0
    )  # Chosen such that method has maximum stability

    _tableau_a = np.array([[_coefficient, 0.0], [1.0 - _coefficient, _coefficient]])
    _tableau_b = np.array([1.0 - _coefficient, _coefficient])
    _tableau_c = np.array([_coefficient, 1.0])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self):
        DiagonallyImplicitRungeKuttaMethod.__init__(self)


class CrouzeixDIRK23(DiagonallyImplicitRungeKuttaMethod):

    _name = "Crouzeix's two-stage third order DIRK method "

    _coefficient_diag = 0.5 + np.sqrt(3) / 6

    _tableau_a = np.array(
        [[_coefficient_diag, 0.0], [-np.sqrt(3) / 3, _coefficient_diag]]
    )
    _tableau_b = np.array([0.5, 0.5])
    _tableau_c = np.array([_coefficient_diag, 0.5 - np.sqrt(3) / 6])

    _n_stages = 2
    _convergence_order = 3.0

    def __init__(self):
        DiagonallyImplicitRungeKuttaMethod.__init__(self)


def solve_ode(ode_solver, ode, t, dt, t_end, verbose=False, TIME_EPS=1e-12):
    """Solve an ordinary differential equation (ODE) using a given ODE solver, time interval and time step size

    :param ode_solver: A Runge-Kutta solver object.
    :type ode_solver: class RungeKuttaMethod or any child class.
    :param ode: Object describing the ODE.
    :type ode: class OrdinaryDifferentialEquation or any child class.
    :param t: Time :math:`t` at which time integration should start.
    :type t: float
    :param dt: Time step size :math:`dt` to use for time integration.
    :type dt: float
    :param t_end: Time :math:`t_{\\mathrm{end}}` at which time integration should end.
    :type t_end: float
    :param verbose: Flag indicating whether the solving procedure should be verbose. If set to `True` intermediate solutions and steps (mainly) of the nonlinear solver will be pritned to screen., defaults to False
    :type verbose: bool, optional
    :param TIME_EPS: Defines how close to :math:`t_{\\mathrm{end}}` the simulation should end., defaults to 1e-12
    :type TIME_EPS: float, optional
    :return: Tuple containing solution at all time steps :math:`t_{i}`, all corresponding times :math:`t_{i}` and the number of needed time steps
    :rtype: tuple( Numpy array, list, integer )
    """
    time_arr, y_arr = [], []
    y_arr.append(ode.get_initial_condition())
    time_arr.append(t)
    y_local = np.copy(ode.get_initial_condition())
    i_steps = 0
    while t < t_end - TIME_EPS:
        if verbose:
            print(f"t={t}: Solve for {t+dt}")
        y_local, t = ode_solver.step(ode, y_local, t, dt, verbose=verbose)
        y_arr.append(y_local)
        time_arr.append(t)
        i_steps += 1
        if t + dt > t_end:
            dt = t_end - t

    return np.array(y_arr), time_arr, i_steps


def run_convergence_test(
    ode_solver, ode, t0, dt0, t_end, n_refinements, expected_order, verbose=False
):
    """Solve a ordinary differential equation (ODE) for a given ODE solver and number of time step refinements

    :param ode_solver: A Runge-Kutta solver object.
    :type RungeKuttaMethod: class RungeKuttaMethod or any child class.

    :param ode: Object describing the ODE.
    :type ode: class OrdinaryDifferentialEquation or any child class.

    :param t0: Initial time from which the integration starts. Usually t0 is zero
               and t0 has to match the initial conditions.
    :type t0: float

    :param dt0: Initial time step size. The time step size is evenly refined (halved).
    :type dt0: float

    :param t_end: The time for which the integration shall stop.
    :type t_end: float

    :param n_refinements: The number of times the time step size should be halved.
    :type n_refinements: integer

    :param expected_order: The time for which the integration shall stop.
    :type expected_order: integer

    :return: Returns the evaluated ODE and the corresponding time as lists.
    :rtype: tuple[list, list]
    """
    errors = []
    time_step_sizes = []
    print(f"Running convergence tests for:\n{ode_solver.get_name()}")

    print("# Steps,         dt,        error, conv. rate,  expect. conv. rate")
    for i_refinement in range(0, n_refinements):
        dt = dt0 * 2 ** (-i_refinement)
        y, _, n_steps = solve_ode(ode_solver, ode, t0, dt, t_end, verbose=verbose)
        error = np.linalg.norm(y[-1] - ode.exact_solution(t_end))
        time_step_sizes.append(dt)
        errors.append(error)
        if i_refinement > 0:
            print(
                f"{n_steps:7d}, {dt:2.5e}, {errors[-1]:2.6e}, {np.log(errors[-2]/errors[-1])/np.log( time_step_sizes[-2] / time_step_sizes[-1]):2.4e}, {expected_order:2.4e}"
            )
        else:
            print(
                f"{n_steps:7d}, {dt:2.5e}, {errors[-1]:2.6e},        ---, {expected_order:2.4e}"
            )

    print("")
    return errors, time_step_sizes
