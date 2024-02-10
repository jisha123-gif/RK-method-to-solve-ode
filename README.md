# RK-method-to-solve-ode
#Rungikutta method to solve ode
""" Author Jisha.CR"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.ticker import ScalarFormatter, LogLocator
def runge_kutta_fourth_order(f, t0, y0, t_end, h):
    """
    Solve a first-order ordinary differential equation dy/dt =y using the fourth-order Runge-Kutta method.

    Parameters:
        f: The function representing the first-order ordinary differential equation dy/dt = f(t, y).
        t0: Initial value of the independent variable.
        y0:Initial value(s) of the dependent variable(s).
        t_end: final value
        h: step size
    """
    N = int((t_end - t0) / h) #N
    t_values = np.linspace(t0, t_end, N + 1)
    y_values = np.zeros(N + 1)
    y_values[0] = y0

    for i in range(N):
        t = t_values[i]
        y = y_values[i]
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        

    return t_values, y_values
def runge_kutta_second_order(f, t0, y0, t_end, h):
    """
    Solve a second-order ordinary differential equation using the fourth-order Runge-Kutta method.
    Solve a first-order ordinary differential equation dy/dt=y using the fourth-order Runge-Kutta method.

    Parameters:
        f: The function representing the first-order ordinary differential equation dy/dt = f(t, y).
        t0: Initial value of the independent variable.
        y0:Initial value(s) of the dependent variable(s).
        t_end: final value
        h: step size
    """
    N = int((t_end - t0) / h)
    t_values = np.linspace(t0, t_end, N + 1)
    y_values = np.zeros(N + 1)
    y_values[0] = y0

    for i in range(N):
        t = t_values[i]
        y = y_values[i]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        y_values[i + 1] = y + k2

    return t_values, y_values
# Example usage:
# Define your function representing the ODE dy/dt = f(t, y)=y 
def f(t, y):
    return y - t*0

# Define initial conditions and parameters
t0 = 0
t_end = 2
y0 = 0.5
h = 0.2

# Solve the ODE using the Runge-Kutta method
#t_values, y_values = runge_kutta_second_order(f, t0, y0, t_end, h)
t_values, y_values = runge_kutta_fourth_order(f, t0, y0, t_end, h)
#############################################################################################################
# Display the results
for t, y in zip(t_values, y_values):
    print(f"t = {t}, y = {y}")
plt.plot(t_values, y_values, label='Solution')

# Fit a linear function to the numerical solution


#plt.gca().yaxis.set_major_formatter(ScalarFormatter())
#plt.gca().yaxis.set_minor_formatter(NullFormatter())
#plt.gca().yaxis.set_major_formatter(ScalarFormatter())
#plt.gca().yaxis.set_minor_locator(LogLocator(subs=np.arange(-2, 10)))
#plt.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt=y using  fourth order Runge-Kutta Method')
#  # Set logarithmic scale for the y-axis
##plt.grid(False)
plt.legend()
plt.show()
def exact_solution(t):
    return 0.5 * np.exp(t)

# Define initial conditions and parameters
t0 = 0
t_end = 3
y0 = 0.5
h = 0.3
# Calculate the exact solution for error calculation
exact_values = exact_solution(t_values)
###########################################################################################################
# Plot the numerical solution and the exact solution
plt.plot(t_values, y_values, label='Numerical Solution')
plt.plot(t_values, exact_values, label='Exact Solution', linestyle='dashed')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Numerical and Exact Solutions of ODE')
#plt.grid(False)
plt.legend()
plt.show()
# Calculate and plot the error
###########################################################################################################
error = np.abs(y_values - exact_values)


linear_fit_coeffs = np.polyfit(t_values, error, 1)
#linear_fit = np.exp(linear_fit_coeffs[1]) * np.exp(linear_fit_coeffs[0] * t_values)
plt.plot(t_values, error, label='Absolute Error')
#plt.plot(t_values, linear_fit, label='Linear Fit', linestyle='dotted')
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title('Absolute Error of Numerical Solution')
plt.yscale('log')
#plt.grid(False)
plt.legend()
plt.show()
######################################################################################
#n_values =[2/0.1,2/0.2,2/0.4,2/0.8,2/1]
# Initialize arrays to store errors and orders of accuracy

#n_values =[(t_end - t0)/0.1, (t_end - t0)/0.2, (t_end - t0)/0.4, (t_end - t0)/0.8, (t_end - t0)/1]
#h_values = [0.01,0.02, 0.04, 0.001,0.0002,0.004]  # Different step sizes to evaluate second order
h_values = [0.01,0.02, 0.04, 0.001,0.002,0.004]  # Different step sizes to evaluate second order
print(h_values)
# Initialize arrays to store errors and orders of accuracy
errors = np.zeros(len(h_values))
orders = np.zeros(len(h_values))
step_sizes = np.zeros(len(h_values))
# Solve the ODE using the second-order Runge-Kutta method for different step sizes
for i, h in enumerate(h_values):
    t_values, y_values = runge_kutta_fourth_order(f, t0, y0, t_end, h)
    #t_values, y_values = runge_kutta_fourth_order(f, t0, y0, t_end, h)
    exact_values = exact_solution(t_values)
    errors[i] = np.max(np.abs(y_values - exact_values))
    step_sizes[i] = h
    if i > 0:
        orders[i] = np.log(errors[i-1] / errors[i]) / np.log(h_values[i-1] / h)

# Plot order of accuracy table
table_data = np.column_stack((h_values, errors, orders))
col_labels = ['Step Size (h)', 'Error', 'Order of Accuracy']
row_labels = [''] * len(h_values)
plt.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels, loc='center')
plt.axis('off')
plt.show()
#################################

# Perform linear regression on logarithm of errors and step sizes
coefficients = np.polyfit(np.log(step_sizes), np.log(errors), 1)

# Extract slope and intercept from the coefficients
slope = coefficients[0]
intercept = coefficients[1]

# Plot the errors and linear fit
plt.scatter(np.log(step_sizes), np.log(errors), label='Errors')
plt.plot(np.log(step_sizes), slope * np.log(step_sizes) + intercept, color='red', label='Linear Fit')

plt.xlabel('Log(Step Size)')
plt.ylabel('Log(Error)')
plt.title('Linear Fit for Error')

plt.legend()
plt.show()

# Print the slope of the linear fit (order of accuracy)
print("Slope (Order of Accuracy):", slope)
