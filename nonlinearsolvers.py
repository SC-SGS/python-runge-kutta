#!/usr/bin/python3

import numpy as np

TOL_EPS = 1e-15

class NonlinearSolverException(Exception):
    pass


def damped_newton_raphson(
    f, df, x0, max_iter=40, tol_rel=1e-12, tol_abs=1e-14, verbose=False
):
    x = x0.copy()
    # print(x, x0)

    residual_abs = np.linalg.norm(f(x), np.inf) + TOL_EPS

    residual_abs_0 = residual_abs
    residual_rel = residual_abs / residual_abs_0

    if verbose:
        print(f"Rel. residual, Abs. residual")
        print(f"{residual_rel:2.6e}, {residual_abs:2.6e}")
    # for i_iter in range(0, max_iter):
    # while ( n_iter < max_iter and ( residual_abs > tol_abs or residual_rel > tol_rel) ):
    n_iter = 0

    damping_factor_min = 1e-4

    while n_iter < max_iter and (residual_abs > tol_abs or residual_rel > tol_rel):
        if verbose:
            print(f"x: {x}")
            print(f"f(x): {f(x)}")
            print(f"df(x): {df(x)}")

        jacobian = df(x)
        rhs = -1.0 * f(x)

        s = np.linalg.solve(jacobian, rhs)

        damping_factor = 1.0
        damping_coefficient = 1.0 - damping_factor / 4.0

        x_guess = x + damping_factor * s
        x_guess_residual_abs = np.linalg.norm(f(x_guess), np.inf)
        x_guess_residual_rel = x_guess_residual_abs / residual_abs_0

        while (
            x_guess_residual_abs > damping_coefficient * residual_abs
            and x_guess_residual_abs > tol_abs
            and x_guess_residual_rel > tol_rel
            and damping_factor > damping_factor_min
        ):
            damping_factor *= 0.5
            x_guess = x + damping_factor * s

            damping_coefficient = 1.0 - damping_factor / 4.0
            x_guess_residual_abs = np.linalg.norm(f(x_guess), np.inf)
            x_guess_residual_rel = x_guess_residual_abs / residual_abs_0
            if verbose:
                print(f"damping_factor: {damping_factor}")

        if damping_factor <= damping_factor_min:
            raise NonlinearSolverException(
                f"Error: Damping factor below threshold ( {damping_factor:1.6e} <= {damping_factor_min:1.6e} )"
            )
            # print()
            # sys.exit(1)

        x = x_guess
        # print(f"s: {s}, x: {x}")

        residual_abs = x_guess_residual_abs
        residual_rel = x_guess_residual_rel

        if verbose:
            print(f"{residual_rel:2.6e}, {residual_abs:2.6e}")

        n_iter += 1
        if n_iter > max_iter:
            raise NonlinearSolverException(
                f"Error: Newton solver did not converge within {max_iter} iterations"
            )

    return x


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
