#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import runge_kutta as rk
import ordinary_differential_equations as ode


def run_convergence_test(ode_solver, y0, t0, dt0, t_end, n_refinements, expected_order):
    """Solve a problem for a given ODE solver and number of time step refinements

    :param ode_solver:
    :type ode_solver: RungeKuttaSolverExplicit
    """
    errors = []
    time_step_sizes = []
    print(f"Running convergence tests for:\n{ode_solver.get_name()}")

    print("         dt,        error, convergence rate, expected conv. rate")
    for i_refinement in range(0, n_refinements):
        dt = dt0 * 2 ** (-i_refinement)
        y, time_arr = rk.solve_ode(ode_solver, y0, t0, dt, t_end, verbose=False)
        error = np.linalg.norm(y[-1] - ode_solver.get_problem().exact_solution(t_end))
        time_step_sizes.append(dt)
        errors.append(error)
        if i_refinement > 0:
            print(
                f"{dt:2.5e}, {errors[-1]:2.6e}, {np.log(errors[-2]/errors[-1])/np.log( time_step_sizes[-2] / time_step_sizes[-1]):2.4e}, {expected_order:2.4e}"
            )
        else:
            print(f"{dt:2.5e}, {errors[-1]:2.6e},        ---, {expected_order:2.4e}")

    print("")
    return errors, time_step_sizes

if __name__ == "__main__":

    ode_problem = ode.SimpleODE()

    t = 0
    dt = 2 ** -4
    t_end = 4

    fig, ax = plt.subplots()
    plt.title(f"Errors over time step size")

    for ode_solver, marker in [
        [rk.ExplicitEuler(ode_problem), "o"],
        [rk.ExplicitImprovedEuler(ode_problem), "v"],
        [rk.Heun(ode_problem), "^"],
        [rk.ClassicalRungeKutta(ode_problem), "s"],
    ]:

        errors, time_step_sizes = run_convergence_test(
            ode_solver=ode_solver,
            y0=np.array([1.0]),
            t0=t,
            dt0=dt,
            t_end=t_end,
            n_refinements=4,
            expected_order=ode_solver.get_convergence_order(),
        )
        ax.loglog(time_step_sizes, errors, label=f"{ode_solver._name}", marker=marker, markersize=8)

    ax.grid()
    plt.xlabel('Time step size $dt$')
    plt.ylabel('Error $|y(t_{end})-y^N|$')

    plt.legend()
    plt.show()
