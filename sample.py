#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rungekuttamethods as rk
import ordinarydifferentialequations as ode

if __name__ == "__main__":

    ode_problem = ode.SimpleODE()

    t0 = 0
    dt = 2 ** -4
    t_end = 4
    y0=np.array([1.0])

    fig, ax = plt.subplots()
    plt.title(f"Errors over time step size")

    for ode_solver, marker in [
        [rk.ExplicitEuler(ode_problem), "o"],
        [rk.ExplicitImprovedEuler(ode_problem), "v"],
        [rk.Heun(ode_problem), "^"],
        [rk.ClassicalRungeKutta(ode_problem), "s"],
    ]:
        y0=np.array([1.0])
        y, time_arr = rk.solve_ode(ode_solver, y0, t0, dt, t_end, verbose=False)
        #print( y, time_arr )
        ax.plot(time_arr, y, label=f"{ode_solver.get_name()}")
        print(f"Error for {ode_solver.get_name()}", np.linalg.norm(y[-1] - ode_solver.get_problem().exact_solution(t_end)))

    x = np.linspace(t0, t_end)
    ax.plot(x, ode_problem.exact_solution(x), label="Exact solution");  # Plot some data on the axes.

    ax.grid()
    plt.xlabel('Time step size $dt$')
    plt.ylabel('Error $|y(t_{end})-y^N|$')

    plt.legend()
    plt.show()