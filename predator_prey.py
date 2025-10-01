import numpy as np
import matplotlib.pyplot as plt

# --- Example 6.6: Euler Method for a System of ODEs (Lotka-Volterra) ---

print("--- Running Example 6.6 (Euler Method for Lotka-Volterra) ---")

# Parameters
a = 1.0   # Prey growth rate
b = 0.1   # Predation rate (prey)
c = 0.5   # Predator decay rate
d = 0.075 # Predator growth rate (efficiency)

h_6_6 = 0.05  # Step size (common for such simulations)
t_start_6_6 = 0.0
t_end_6_6 = 50.0 # Simulate for a longer period to see cycles
x0_6_6 = 10.0 # Initial prey population
y0_6_6 = 5.0  # Initial predator population
Y0_6_6 = np.array([x0_6_6, y0_6_6]) # Initial state vector

# System of ODEs function (returns a vector of rates)
def f_6_6(t, Y):
    """
    Right-hand side of the Lotka-Volterra system.
    Y[0] is x (prey), Y[1] is y (predator)
    """
    x, y = Y
    dxdt = a*x - b*x*y
    dydt = -c*y + d*x*y
    return np.array([dxdt, dydt])

# Euler Method for Systems Implementation
def euler_method_system(f, Y0, t_start, t_end, h):
    num_steps = int(np.ceil((t_end - t_start) / h))
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    Y_values = np.zeros((num_steps + 1, len(Y0))) # Store x and y components

    Y_values[0] = Y0

    for i in range(num_steps):
        Y_values[i+1] = Y_values[i] + h * f(t_values[i], Y_values[i])
    return t_values, Y_values

# Run the Euler Method for the system
t_lv, Y_lv_euler = euler_method_system(f_6_6, Y0_6_6, t_start_6_6, t_end_6_6, h_6_6)
x_lv_euler = Y_lv_euler[:, 0]
y_lv_euler = Y_lv_euler[:, 1]

# Plotting Results (Time Series and Phase Portrait)
plt.figure(figsize=(14, 6))

# Time Series Plot
plt.subplot(1, 2, 1)
plt.plot(t_lv, x_lv_euler, 'b-', label='Prey (x)')
plt.plot(t_lv, y_lv_euler, 'g-', label='Predator (y)')
plt.title(f'Example 6.6: Lotka-Volterra Time Series (Euler Method, h={h_6_6})')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0) # Populations cannot be negative

# Phase Portrait Plot
plt.subplot(1, 2, 2)
plt.plot(x_lv_euler, y_lv_euler, 'm-', label='Trajectory')
plt.plot(x0_6_6, y0_6_6, 'ro', markersize=8, label='Initial Condition')
plt.title('Example 6.6: Lotka-Volterra Phase Portrait (Euler Method)')
plt.xlabel('Prey Population (x)')
plt.ylabel('Predator Population (y)')
plt.legend()
plt.grid(True)
plt.xlim(left=0) # Populations cannot be negative
plt.ylim(bottom=0) # Populations cannot be negative

plt.tight_layout()
plt.show()

# Print some numerical values
print("\n--- Example 6.6 Numerical Results (Euler Method for Lotka-Volterra) ---")
print(f"{'t':<5} {'Prey (x)':<15} {'Predator (y)':<15}")
for i in range(0, len(t_lv), int(len(t_lv)/10)): # Print a few points
    print(f"{t_lv[i]:<5.2f} {x_lv_euler[i]:<15.8f} {y_lv_euler[i]:<15.8f}")
print("...")
print(f"{t_lv[-1]:<5.2f} {x_lv_euler[-1]:<15.8f} {y_lv_euler[-1]:<15.8f}")