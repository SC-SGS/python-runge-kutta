Runge-Kutta solvers for ordinary differential equations
=======================================================

|Test badge| |Style check|

`rungekutta` is a Python module that provides a framework for defining ordinary differential equations (ODEs) and solving these differential equations using Runge-Kutta methods.

Dependencies
------------

-  Python 3
-  Numpy
-  matplotlib for examples

Installation
------------

The module can be installed via ``pip``. If you want to develop this package, install the module in "editable" mode, i.e., add the ``-e`` flag to the ``install`` directive below

1. Clone the repository from GitHub and install it

    .. code::
        git clone https://github.com/ajaust/python-runge-kutta
        pip install --user python-runge-kutta

2. Install the module directly from GitHub

    .. code::
        pip instal --user https://github.com/ajaust/python-runge-kutta/archive/master.zip

The installed module is available under the name ``rungekutta``.

Usage
-----

This example shows how to to solve one of the provided ODE examples with an already implemented
Runge-Kutta method. We use the explicit Euler method:

.. code::python
    import matplotlib.pyplot as plt
    import numpy as np

    from rungekutta import rungekuttamethods as rk
    from rungekutta import ordinarydifferentialequations as odes

    # Start time
    t0 = 0
    # Time step size
    dt = 2**-4
    # End time
    t_end = 4

    # Initialize a simple ode y'(t) = y - 2*sin(t) as problem
    ode = odes.SimpleODE()
    # Initialize explicit Euler method as solver
    ode_solver = rk.ExplicitEuler()
    # Solve the
    y, time_arr, _ = rk.solve_ode(ode_solver, ode, t0, dt, t_end, verbose=False)
    y = y.flatten()

    # Plot solution using matplotlib
    fig, ax = plt.subplots()
    plt.title(f"Solution of a simple ODE")
    ax.plot(time_arr, y, label=f"{ode_solver.get_name()}", marker="o")

    # Plot the exact solution of the problem for comparison
    x = np.linspace(t0, t_end)
    ax.plot(x, ode.exact_solution(x), label="Exact solution")

    ax.grid()
    plt.xlabel("Time $t$")
    plt.ylabel("Solution $y(t)$")

    ax.set_xlim([t0 - 0.1, t_end + 0.1])
    ax.set_ylim([-3.5, 2.0])

    plt.legend()
    plt.show()

For more examples check out the ``examples/`` directory in the repository.

.. |Test badge| image:: https://github.com/ajaust/python-runge-kutta/actions/workflows/tests.yml/badge.svg
.. |Style check| image:: https://github.com/ajaust/python-runge-kutta/actions/workflows/style-check.yml/badge.svg
