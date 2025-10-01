import numpy as np
import matplotlib.pyplot as plt

# --- Example 6.3: Improved Euler Method ---

print("--- Running Example 6.3 (Improved Euler Method) ---")

# Parameters
h_6_3 = 0.1  # Step size (as in the textbook example)
t_start_6_3 = 0.0
t_end_6_3 = 2.0
y0_6_3 = 1.0  # Initial condition y(0) = 1

# ODE Function (same as 6.2)
def f_6_3(t, y):
    """Right-hand side of the ODE dy/dt = y"""
    return y

# Exact Solution (same as 6.2)
def exact_solution_6_3(t):
    """Exact solution y(t) = e^t"""
    return np.exp(t)

# Improved Euler Method Implementation
def improved_euler_method(f, y0, t_start, t_end, h):
    num_steps = int(np.ceil((t_end - t_start) / h))
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    y_values[0] = y0

    for i in range(num_steps):
        t_n = t_values[i]
        y_n = y_values[i]
        t_n_plus_1 = t_values[i+1]

        # Euler's prediction (predictor step)
        y_star = y_n + h * f(t_n, y_n)

        # Improved Euler formula (corrector step)
        y_values[i+1] = y_n + h * (f(t_n, y_n) + f(t_n_plus_1, y_star)) / 2
    return t_values, y_values

# Run the Improved Euler Method
t_improved_euler, y_improved_euler = improved_euler_method(f_6_3, y0_6_3, t_start_6_3, t_end_6_3, h_6_3)
y_exact_6_3 = exact_solution_6_3(t_improved_euler)

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(t_improved_euler, y_exact_6_3, 'r--', label='Exact Solution $e^t$')
plt.plot(t_improved_euler, y_improved_euler, 'g-x', markersize=4, label=f'Improved Euler ($h={h_6_3}$)')
plt.title('Example 6.3: Improved Euler Method for dy/dt = y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print Numerical Results
print("--- Example 6.3 Numerical Results (Improved Euler Method) ---")
print(f"{'t':<5} {'Improved Euler y':<20} {'Exact y':<20} {'Error':<20}")
for i in range(len(t_improved_euler)):
    print(f"{t_improved_euler[i]:<5.1f} {y_improved_euler[i]:<20.10f} {y_exact_6_3[i]:<20.10f} {abs(y_exact_6_3[i] - y_improved_euler[i]):<20.10f}")
print("\n" + "="*80 + "\n")