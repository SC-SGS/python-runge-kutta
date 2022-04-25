#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rungekuttamethods as rk
import ordinarydifferentialequations

if __name__ == "__main__":

    ode = ordinarydifferentialequations.SimpleODE()

    t = 0
    dt = 2**-4
    t_end = 4

    fig, ax = plt.subplots()
    plt.title(f"Errors over time step size")

    for ode_solver, marker in [
        [rk.ImplicitEuler(), "o"],
        [rk.ImplicitTrapezoidalRule(), "v"],
        [rk.DIRK22(), "^"],
        [rk.CrouzeixDIRK23(), "s"],
    ]:

        errors, time_step_sizes = rk.run_convergence_test(
            ode_solver=ode_solver,
            ode=ode,
            t0=t,
            dt0=dt,
            t_end=t_end,
            n_refinements=4,
            expected_order=ode_solver.get_convergence_order(),
            verbose=False,
        )
        ax.loglog(
            time_step_sizes,
            errors,
            label=f"{ode_solver._name}",
            marker=marker,
            markersize=8,
        )

    ax.grid()
    plt.xlabel("Time step size $dt$")
    plt.ylabel("Error $|y(t_{end})-y^N|$")

    plt.legend()
    plt.show()
