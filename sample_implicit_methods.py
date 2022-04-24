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

    fig, ax = plt.subplots()
    plt.title(f"Errors over time step size")

    for ode_solver, marker in [
        [rk.ExplicitEuler(ode_problem), "+"],
        [rk.ImplicitEuler(ode_problem), "o"],
        [rk.ImplicitTrapezoidalRule(ode_problem), "v"],
        [rk.DIRK22(ode_problem), "^"],
        [rk.CrouzeixDIRK23(ode_problem), "s"],
        #[rk.ExplicitImprovedEuler(ode_problem), "v"],
        #[rk.Heun(ode_problem), "^"],
        #[rk.ClassicalRungeKutta(ode_problem), "s"],
    ]:
        y, time_arr = rk.solve_ode(ode_solver, t0, dt, t_end, verbose=False)
        y = y.flatten()
        # print( y, time_arr )
        ax.plot(time_arr, y, label=f"{ode_solver.get_name()}", marker=marker)
        print(
            f"Error for {ode_solver.get_name()}",
            np.linalg.norm(y[-1] - ode_solver.get_problem().exact_solution(t_end)),
        )

    x = np.linspace(t0, t_end)
    ax.plot(x, ode_problem.exact_solution(x), label="Exact solution")

    ax.grid()
    plt.xlabel("Time $t$")
    plt.ylabel("Solution $y(t)$")

    plt.legend()
    plt.show()
