#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import runge_kutta
import ordinary_differential_equations as ode

TIME_EPS=1e-12

#asdf = runge_kutta.RungeKuttaSolver()

def solve_ode( ode_integrator, y, t, dt, t_end):
    ode_integrator.report()
    # Setting initial value
    time_arr, y_arr = [], []
    y_arr.append(y[0])
    time_arr.append(t)
    print(y_arr)
    while t < t_end - TIME_EPS:
        y, t = ode_integrator.step( y, t, dt )
        y_arr.append(y[0])
        time_arr.append(t)

    return y_arr, time_arr

ode_lambda = -1.0
#ode_problem = runge_kutta.ExponentialFunction(lam=ode_lambda)
ode_problem = ode.SimpleODE()

explicit_euler_solver = runge_kutta.ExplicitEuler(ode_problem)
explicit_improved_euler_solver = runge_kutta.ExplicitImprovedEuler(ode_problem)
explicit_heun_solver = runge_kutta.Heun(ode_problem)
explicit_rk4_solver = runge_kutta.ClassicalRungeKutta(ode_problem)
#explicit_rk4_solver.report()


#for n in [-4, -5, -6, -7]:
for n in [-4]:
    print(f"Solve problem for dt = 2**{n}")
    t = 0
    dt = 2**n
    t_end = 2**n

    y = np.array( [1.] )
    y_expl_euler, time_arr = solve_ode( explicit_improved_euler_solver, y, t, dt, t_end )
    #print( time_arr, y_expl_euler )
    error = np.linalg.norm( y_expl_euler[-1] - ode_problem.exact_solution(t_end) )
    print(f"Error {explicit_improved_euler_solver._name}: {error}")

#y = np.array( [1.] )
#y_expl_imprv_euler, time_arr = solve_ode( explicit_improved_euler_solver, y, t, dt, t_end )
#print( time_arr, y_expl_imprv_euler )
#
#y = np.array( [1.] )
#y_heun, time_arr = solve_ode( explicit_heun_solver, y, t, dt, t_end )
#print( time_arr, y_heun )
#
#y = np.array( [1.] )
#y_rk4, time_arr = solve_ode( explicit_rk4_solver, y, t, dt, t_end )
#print( time_arr, y_rk4 )

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(time_arr, y_expl_euler, label="Explicit Euler");  # Plot some data on the axes.
#ax.plot(time_arr, y_expl_imprv_euler, label="Improved Euler");  # Plot some data on the axes.
#ax.plot(time_arr, y_heun, label="Heun");  # Plot some data on the axes.
#ax.plot(time_arr, y_rk4, label="RK4");  # Plot some data on the axes.

x = np.linspace(t, t_end)
#ax.plot(x, 5*np.exp(ode_lambda*x), label="Exact solution");  # Plot some data on the axes.
ax.plot(x, ode_problem.exact_solution(x), label="Exact solution");  # Plot some data on the axes.

#plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
#plt.plot(x, x**2, label='quadratic')  # etc.
#plt.plot(x, x**3, label='cubic')
#plt.xlabel('x label')
#plt.ylabel('y label')
#plt.title("Simple Plot")
plt.legend();

plt.show()