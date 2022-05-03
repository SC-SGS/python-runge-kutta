Code examples using rungekutta
==============================

Please note that you can find more examples in the ``examples/`` directory of the GitHub repository of this module.

Solving a simple ordinary differential equation
-----------------------------------------------

This examples solves the ODE defined by :py:class:`SimpleODE`.

.. literalinclude:: ../../examples/example_explicit_methods.py
   :language: python
   :linenos:

You can also easily rewrite the script to use implicit Runge-Kutta methods instead:

.. literalinclude:: ../../examples/example_implicit_methods.py
   :language: python
   :linenos: