#!/usr/bin/python3
"""Nonlinear solver module

This module contains methods for solving nonlinear systems of equations and is used by the :py:mod:`rungekutta` module.
"""
import numpy as np

TOL_EPS = 1e-15


class NonlinearSolverException(Exception):
    """Class for expections concerning nonlinear solvers

    This exception is thrown if a nonlinear solver is unable to converge.
    """

    pass


def damped_newton_raphson(
    f, df, x0, max_iter=20, tol_rel=1e-10, tol_abs=1e-10, verbose=False
):
    """Solve a given (non)linear sytem of equations and solve it using a damped Newton-Raphson method. The problem

    .. math::
        f(x) = 0

    is solved iteratively by solving

    .. math::
        -f(x^k)/f'(x^k) = s^k

    and updating

    .. math::
        x^{k+1} = x^k + \lambda \cdot s^k

    where :math:`\lambda` is the damping factor. The unknown may be a vector, i.e, :math:`x \\in \\mathbb{R}^{n}$.
    The iterative procedure is stopped if any of the following conditions is fulfilled:

        1. `max_iter` iterations of the algorithm have been carried.
        2. The relative residual is small enough, i.e.,
            .. math::
               \lVert f(x^k) / f(x^0) \\rVert_{\infty} < \mathrm{tol_rel}
        3. The absolute residual is small enough, i.e.,
            .. math::
                \lVert f(x^k) \\rVert_\infty < \mathrm{tol_abs}

    The implementation is based on W. Dahmen and A. Reusken: "Numerik für Ingenieure und Naturwissenschaftler", 2008, `doi:10.1007/978-3-540-76493-9 <https://dx.doi.org/10.1007/978-3-540-76493-9>`_

    :param f: Function to evaluate in order to get the residual.
    :type f: function
    :param df:  Function to evaluate in order to get the jacobian of f.
    :type df: function that should return a numpy array of shape (n,n)
    :param x0: Initial guess for solving the nonlinear problem f(x)=0.
    :type x0: Numpy array of dimension (n,1)
    :param max_iter:  Newton iterations to carry out at most., defaults to 20
    :type max_iter: int, optional
    :param tol_rel: Relative residual of the solution update to be reached., defaults to 1e-10
    :type tol_rel: float, optional
    :param tol_abs: Absolute residual of the solution update to be reached., defaults to 1e-10
    :type tol_abs: float, optional
    :param verbose:  Flag indicating whether the solving procedure should be verbose. If set to `True`, many intermediate quantities and the computed residuals will be printed to the screen., defaults to False
    :type verbose: bool, optional
    :raises NonlinearSolverException: Thrown if the damped Newton methods diverges either due to the damping factor being decreased too much or when the maximum number of Newton steps `max_iter` is reached.
    :return: Returns :math:`x` with :math:`f(x) \\approx 0` and the number of Newton iterations carried out.
    :rtype: tuple[ np.array(n, 1), int ]
    """
    x = np.copy(x0)

    residual_abs = np.linalg.norm(f(x), np.inf) + TOL_EPS

    residual_abs_0 = residual_abs
    residual_rel = residual_abs / residual_abs_0

    if verbose:
        print(f"Rel. residual, Abs. residual")
        print(f"{residual_rel:2.6e}, {residual_abs:2.6e}")
    n_iter = 0

    damping_factor_min = 1e-4

    while n_iter < max_iter and (residual_abs > tol_abs or residual_rel > tol_rel):
        if verbose:
            print(f"x: {x}")
            print(f"f(x): {f(x)}")
            print(f"df(x): {df(x)}")

        jacobian = df(x)
        rhs = -1.0 * f(x)

        if verbose:
            print(f"Jacobian: {jacobian}")
            print(f"rhs: {rhs}")

        s = np.linalg.solve(jacobian, rhs)

        damping_factor = 1.0
        damping_coefficient = 1.0 - damping_factor / 4.0

        x_guess = x + damping_factor * s
        x_guess_residual_abs = np.linalg.norm(f(x_guess), np.inf)
        x_guess_residual_rel = x_guess_residual_abs / residual_abs_0

        while x_guess_residual_abs > damping_coefficient * residual_abs and (
            x_guess_residual_rel > tol_rel and x_guess_residual_abs > tol_abs
        ):
            damping_factor *= 0.5
            if damping_factor <= damping_factor_min:
                raise NonlinearSolverException(
                    f"Error: Damping factor below threshold ( {damping_factor:1.6e} <= {damping_factor_min:1.6e} )"
                )

            x_guess = x + damping_factor * s

            damping_coefficient = 1.0 - damping_factor / 4.0
            x_guess_residual_abs = np.linalg.norm(f(x_guess), np.inf)
            x_guess_residual_rel = x_guess_residual_abs / residual_abs_0
            if verbose:
                print(f"x_guess: {x_guess}, s: {s}")
                print(f"damping_factor: {damping_factor:1.6e}")
                print(
                    f"x_guess_residual_abs: {x_guess_residual_abs}, x_guess_residual_rel: {x_guess_residual_rel}"
                )

        x = x_guess

        residual_abs = x_guess_residual_abs
        residual_rel = x_guess_residual_rel

        if verbose:
            print(f"{residual_rel:2.6e}, {residual_abs:2.6e}")

        n_iter += 1
        if n_iter > max_iter:
            raise NonlinearSolverException(
                f"Error: Newton solver did not converge within {max_iter} iterations"
            )

    return x, n_iter


if __name__ == "__main__":
    x0 = np.array([0, 0])
    f = lambda x: np.array(
        [6 * x[0] - np.cos(x[0]) - 2 * x[1], 8 * x[1] - x[0] * x[1] ** 2 - np.sin(x[0])]
    )
    df = lambda x: np.array(
        [[6.0 + np.sin(x[0]), -2], [-x[1] ** 2 - np.cos(x[0]), 8.0 - 2 * x[0] * x[1]]]
    )

    x = damped_newton_raphson(f, df, x0, 20, verbose=True)
    print(x)  # Solution should be close to (0.17133369, 0.02132175)
