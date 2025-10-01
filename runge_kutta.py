import numpy as np
import matplotlib.pyplot as plt

# --- Example 6.4: Runge-Kutta 4th Order (RK4) ---

print("--- Running Example 6.4 (RK4 Method) ---")

# Parameters
h_6_4 = 0.1  # Step size (as in the textbook example)
t_start_6_4 = 0.0
t_end_6_4 = 2.0
y0_6_4 = 1.0  # Initial condition y(0) = 1

# ODE Function (same as 6.2 and 6.3)
def f_6_4(t, y):
    """Right-hand side of the ODE dy/dt = y"""
    return y

# Exact Solution (same as 6.2 and 6.3)
def exact_solution_6_4(t):
    """Exact solution y(t) = e^t"""
    return np.exp(t)

# Runge-Kutta 4th Order (RK4) Implementation
def rk4_method(f, y0, t_start, t_end, h):
    num_steps = int(np.ceil((t_end - t_start) / h))
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    y_values[0] = y0

    for i in range(num_steps):
        t_n = t_values[i]
        y_n = y_values[i]

        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h/2 * k1)
        k3 = f(t_n + h/2, y_n + h/2 * k2)
        k4 = f(t_n + h, y_n + h * k3)

        y_values[i+1] = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return t_values, y_values

# Run the RK4 Method
t_rk4, y_rk4 = rk4_method(f_6_4, y0_6_4, t_start_6_4, t_end_6_4, h_6_4)
y_exact_6_4 = exact_solution_6_4(t_rk4)

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(t_rk4, y_exact_6_4, 'r--', label='Exact Solution $e^t$')
plt.plot(t_rk4, y_rk4, 'c-^', markersize=4, label=f'RK4 Method ($h={h_6_4}$)')
plt.title('Example 6.4: RK4 Method for dy/dt = y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print Numerical Results
print("--- Example 6.4 Numerical Results (RK4 Method) ---")
print(f"{'t':<5} {'RK4 y':<20} {'Exact y':<20} {'Error':<20}")
for i in range(len(t_rk4)):
    print(f"{t_rk4[i]:<5.1f} {y_rk4[i]:<20.10f} {y_exact_6_4[i]:<20.10f} {abs(y_exact_6_4[i] - y_rk4[i]):<20.10f}")
print("\n" + "="*80 + "\n")