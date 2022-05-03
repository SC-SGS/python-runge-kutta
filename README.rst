Runge-Kutta solvers for ordinary differential equations
=======================================================

|Test badge| |Style check|

`rungekutta` is a Python module that provides a framework for defining ordinary differential equations and solving these differential equations using Runge-Kutta methods.

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

The installed module is available under the name `rungekutta`.

Usage
-----

For more examples check out the `examples/` directory in the repository.

.. |Test badge| image:: https://github.com/ajaust/python-runge-kutta/actions/workflows/tests.yml/badge.svg
.. |Style check| image:: https://github.com/ajaust/python-runge-kutta/actions/workflows/style-check.yml/badge.svg
