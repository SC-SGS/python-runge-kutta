#!/usr/bin/env python3

import matplotlib.pyplot as plt

import rungekuttamethods as rk
import ordinarydifferentialequations

if __name__ == "__main__":

    ode = ordinarydifferentialequations.VanDerPol()

    t0 = 0
    dt = (
        2**-5
    )  # Should be at most 2^-5 for explicit Euler for stability, but result will still be poor. 2^-4 works ok for higher order methods.
    t_end = 30

    fig, ax = plt.subplots()
    plt.title(f"Solution of van der Pol equation")

    for ode_solver, marker in [
        [rk.ExplicitEuler(), "o"],
        [rk.ExplicitImprovedEuler(), "v"],
        [rk.Heun(), "^"],
        [rk.ClassicalRungeKutta(), "s"],
    ]:
        y, time_arr, _ = rk.solve_ode(ode_solver, ode, t0, dt, t_end, verbose=False)
        ax.plot(time_arr, y[:, 0], label=f"{ode_solver.get_name()}")

    ax.set_ylim([-2.5, 2.5])
    ax.grid()
    plt.xlabel("Time $t$")
    plt.ylabel("Solution $y(t)$")

    plt.legend()
    plt.show()
