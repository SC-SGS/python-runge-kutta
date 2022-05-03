#!/usr/bin/env python3

"""Example: Implementation of an own Runge-Kutta method

In this example script a new diagonally implicit Runge-Kutta
method is implemented and used to solve the ``SimpleODE``
test case.
"""

import matplotlib.pyplot as plt
import numpy as np

from rungekutta import rungekuttamethods as rk
from rungekutta import ordinarydifferentialequations as odes


class QinZhangDIRK22(rk.DiagonallyImplicitRungeKuttaMethod):

    _name = "Symplectic DIRK22 method by Qin and Zhang"

    _tableau_a = np.array([[0.25, 0.0], [0.5, 0.25]])
    _tableau_b = np.array([0.5, 0.5])
    _tableau_c = np.array([0.25, 0.75])

    _n_stages = 2
    _convergence_order = 2.0

    def __init__(self):
        rk.DiagonallyImplicitRungeKuttaMethod.__init__(self)


if __name__ == "__main__":

    ode = odes.SimpleODE()

    t0 = 0
    dt = 2**-4
    t_end = 4

    fig, ax = plt.subplots()
    plt.title(f"Solution over time using implicit methods")

    ode_solver = QinZhangDIRK22()

    y, time_arr, _ = rk.solve_ode(ode_solver, ode, t0, dt, t_end, verbose=False)
    y = y.flatten()
    ax.plot(time_arr, y, label=f"{ode_solver.get_name()}", marker="o")
    print(
        f"Error for {ode_solver.get_name()}: {np.linalg.norm(y[-1] - ode.exact_solution(t_end)):1.6e}"
    )

    x = np.linspace(t0, t_end)
    ax.plot(x, ode.exact_solution(x), label="Exact solution")

    ax.grid()
    plt.xlabel("Time $t$")
    plt.ylabel("Solution $y(t)$")

    ax.set_xlim([t0 - 0.1, t_end + 0.1])
    ax.set_ylim([-3.5, 2.0])

    plt.legend()
    plt.show()
