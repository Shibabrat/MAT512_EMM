import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define the model as a function
# This function returns the derivatives [dx/dt, dv/dt]
def harmonic_oscillator(t, y, m, k, b):
    """
    Defines the system of first-order ODEs for a damped harmonic oscillator.
    
    Args:
        t (float): Time (required by solve_ivp, but not used in this model).
        y (list or np.array): State vector [position x, velocity v].
        m (float): Mass.
        k (float): Spring constant.
        b (float): Damping coefficient.
        
    Returns:
        list: The derivatives [dx/dt, dv/dt].
    """
    x, v = y  # Unpack the state vector
    dxdt = v
    dvdt = (-b * v - k * x) / m
    return [dxdt, dvdt]

# 2. Set up simulation parameters and initial conditions
m = 1.0      # Mass (kg)
k = 1.5      # Spring constant (N/m)
b = 0.25     # Damping coefficient (Ns/m)

# Initial state: [initial position, initial velocity]
# Let's pull the spring back 5 units and release it from rest.
y0 = [5.0, 0.0]

# Time span for the simulation
t_span = [0, 25]  # Simulate from t=0 to t=25 seconds
# Points in time where we want the solution
t_eval = np.linspace(t_span[0], t_span[1], 500)

# 3. Run the simulation using SciPy's solve_ivp
# 'args' passes the model parameters (m, k, b) to our function
solution = solve_ivp(
    fun=harmonic_oscillator,
    t_span=t_span,
    y0=y0,
    args=(m, k, b),
    t_eval=t_eval,
    dense_output=True
)

# 4. Extract and visualize the results
# The solution object contains the time points and the state at each point
t = solution.t
position = solution.y[0]
velocity = solution.y[1]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, position, label='Position (x)', color='blue')
plt.plot(t, velocity, label='Velocity (v)', color='red', linestyle='--')
plt.title('Damped Simple Harmonic Oscillator Simulation')
plt.xlabel('Time (s)')
plt.ylabel(r'$x(t), v(t)$')
plt.legend()
plt.grid(True)
plt.show()


