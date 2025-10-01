import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Lorenz Equations Function ---
def lorenz_equations(state, t, sigma, rho, beta):
    """
    Defines the Lorenz system of differential equations.
    state = [x, y, z]
    t = time (unused in this autonomous system, but required by some solvers)
    sigma, rho, beta = Lorenz parameters
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# --- RK4 Method for Systems Implementation ---
# This is a general RK4 solver for systems of ODEs
def rk4_system_solver(func, Y0, t_span, h, *args):
    """
    Solves a system of ODEs using the RK4 method.

    Args:
        func: A function f(Y, t, *args) that returns the derivatives dY/dt.
              Y is a numpy array for the state variables.
        Y0: Initial state vector (numpy array).
        t_span: Tuple (t_start, t_end).
        h: Step size.
        *args: Additional arguments to pass to func.

    Returns:
        t_values: Array of time points.
        Y_values: 2D array where Y_values[i, :] is the state vector at t_values[i].
    """
    t_start, t_end = t_span
    num_steps = int(np.ceil((t_end - t_start) / h))
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    Y_values = np.zeros((num_steps + 1, len(Y0)))
    Y_values[0] = Y0

    for i in range(num_steps):
        t_n = t_values[i]
        Y_n = Y_values[i]

        k1 = func(Y_n, t_n, *args)
        k2 = func(Y_n + h/2 * k1, t_n + h/2, *args)
        k3 = func(Y_n + h/2 * k2, t_n + h/2, *args)
        k4 = func(Y_n + h * k3, t_n + h, *args)

        Y_values[i+1] = Y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return t_values, Y_values

# --- Parameters for Lorenz Simulation ---
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

h_lorenz = 0.01  # Step size for integration
t_start_lorenz = 0.0
t_end_lorenz = 50.0  # Simulation duration

# --- Initial Conditions for the "Butterfly Effect" demonstration ---
# Two very slightly different initial conditions
initial_state_1 = np.array([0.0, 1.0, 1.05])
initial_state_2 = np.array([0.0, 1.0, 1.05 + 1e-5]) # Very small perturbation

# --- Run RK4 Solver for both trajectories ---
print("--- Simulating Lorenz System ---")
print(f"Parameters: sigma={sigma}, rho={rho}, beta={beta}")
print(f"Initial State 1: {initial_state_1}")
print(f"Initial State 2 (perturbed): {initial_state_2}")

t_values, trajectory_1 = rk4_system_solver(lorenz_equations, initial_state_1,
                                            (t_start_lorenz, t_end_lorenz), h_lorenz,
                                            sigma, rho, beta)

_, trajectory_2 = rk4_system_solver(lorenz_equations, initial_state_2,
                                    (t_start_lorenz, t_end_lorenz), h_lorenz,
                                    sigma, rho, beta)

# --- Plotting Lorenz Attractor (3D) ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], trajectory_1[:, 2], 'b', label='Trajectory 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor (3D Phase Space)')
plt.legend()
plt.grid(True)
plt.show()

# --- Plotting Butterfly Effect (Time Series Comparison) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
variables = ['X', 'Y', 'Z']

for i in range(3):
    axes[i].plot(t_values, trajectory_1[:, i], 'b', label=f'{variables[i]} (Initial 1)')
    axes[i].plot(t_values, trajectory_2[:, i], 'r--', label=f'{variables[i]} (Initial 2, perturbed)')
    axes[i].set_ylabel(variables[i])
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

axes[0].set_title('Lorenz System: Butterfly Effect (Time Series for Two Close Initial Conditions)')
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

# --- Plotting Differences Over Time ---
differences = np.linalg.norm(trajectory_1 - trajectory_2, axis=1) # Euclidean distance between states

plt.figure(figsize=(12, 6))
plt.plot(t_values, differences, 'k-', label='Euclidean Distance |Trajectory 1 - Trajectory 2|')
plt.yscale('log') # Log scale is often used to show exponential divergence
plt.title('Divergence of Two Nearby Lorenz Trajectories (Log Scale)')
plt.xlabel('Time')
plt.ylabel('Distance (log scale)')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()

print("\n--- Simulation Complete ---")