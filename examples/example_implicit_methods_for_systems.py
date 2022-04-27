#!/usr/bin/env python3

import matplotlib.pyplot as plt

from rungekutta import rungekuttamethods as rk
from rungekutta import ordinarydifferentialequations

if __name__ == "__main__":

    ode = ordinarydifferentialequations.VanDerPol()

    t0 = 0
    # Time step size dt=0.1 should give stable result for implicit methods already, but with varying quality.
    dt = 10**-1

    t_end = 30

    fig, ax = plt.subplots()
    plt.title(f"Solution of van der Pol equation")

    for ode_solver, marker in [
        [rk.ImplicitEuler(), "o"],
        [rk.ImplicitTrapezoidalRule(), "v"],
        [rk.DIRK22(), "^"],
        [rk.CrouzeixDIRK23(), "s"],
    ]:
        y, time_arr, _ = rk.solve_ode(ode_solver, ode, t0, dt, t_end, verbose=False)
        ax.plot(time_arr, y[:, 0], label=f"{ode_solver.get_name()}", marker=marker)

    ax.set_ylim([-2.5, 2.5])
    ax.grid()
    plt.xlabel("Time $t$")
    plt.ylabel("Solution $y(t)$")

    plt.legend()
    plt.show()
