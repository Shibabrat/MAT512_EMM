import numpy as np
import matplotlib.pyplot as plt

# --- Example 6.2: Euler Method ---

print("--- Running Example 6.2 (Euler Method) ---")

# Parameters
h_6_2 = 0.01  # Step size (as in the textbook example)
t_start_6_2 = 0.0
t_end_6_2 = 2.0
y0_6_2 = 1.0  # Initial condition y(0) = 1

# ODE Function
def f_6_2(t, y):
    """Right-hand side of the ODE dy/dt = y"""
    return y

# Exact Solution
def exact_solution_6_2(t):
    """Exact solution y(t) = e^t"""
    return np.exp(t)

# Euler Method Implementation
def euler_method(f, y0, t_start, t_end, h):
    num_steps = int(np.ceil((t_end - t_start) / h)) # Use ceil to ensure t_end is reached or slightly exceeded
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    y_values[0] = y0

    for i in range(num_steps):
        y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i])
    return t_values, y_values

# Run the Euler Method
t_euler, y_euler = euler_method(f_6_2, y0_6_2, t_start_6_2, t_end_6_2, h_6_2)
y_exact_6_2 = exact_solution_6_2(t_euler)

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(t_euler, y_exact_6_2, 'r--', label='Exact Solution $e^t$')
plt.plot(t_euler, y_euler, 'b-o', markersize=4, label=f'Euler Method ($h={h_6_2}$)')
plt.title('Example 6.2: Euler Method for dy/dt = y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print Numerical Results
print("--- Example 6.2 Numerical Results (Euler Method) ---")
print(f"{'t':<5} {'Euler y':<15} {'Exact y':<15} {'Error':<15}")
for i in range(len(t_euler)):
    print(f"{t_euler[i]:<5.1f} {y_euler[i]:<15.8f} {y_exact_6_2[i]:<15.8f} {abs(y_exact_6_2[i] - y_euler[i]):<15.8f}")
print("\n" + "="*80 + "\n")

