import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define the model as a function
# This function returns the derivatives [dx1/dt, dx2/dt]
def rlc_circuit(t, y, L, C):
    """
    Defines the system of first-order ODEs for a RLC circuit example.
    
    Args:
        t (float): Time (required by solve_ivp, but not used in this model).
        y (list or np.array): State vector [current i_L = i_C = i_R, voltage v_C].
        L (float): Inductance.
        C (float): Capacitance.
        
    Returns:
        list: The derivatives [dx1/dt, dx2/dt].
    """
    x1, x2 = y  # Unpack the state vector
    # dx1dt = (-x1**3 - 4*x1 - x2)/L # v-i characteristic function, f(x) = x^3 + 4x
    dx1dt = (-x1**3 + x1 - x2)/L # # v-i characteristic function, f(x) = x^3 - x
    dx2dt = x1/C
    return [dx1dt, dx2dt]

# 2. Set up simulation parameters and initial conditions
L = 1.0      # Inductance 
C = 1/3    # Capacitance 

# Initial state: [initial current, initial voltage]
y0 = [-2.0, -2.0]

# Time span for the simulation
t_span = [0, 50]  # Simulate from t=0 to t=15 seconds
# Points in time where we want the solution
t_eval = np.linspace(t_span[0], t_span[1], 500)

# 3. Run the simulation using SciPy's solve_ivp
# 'args' passes the model parameters (L, C) to our function
solution = solve_ivp(
    fun=rlc_circuit,
    t_span=t_span,
    y0=y0,
    args=(L, C),
    t_eval=t_eval,
    dense_output=True
)

# 4. Extract and visualize the results
# The solution object contains the time points and the state at each point
t = solution.t
current = solution.y[0]
voltage = solution.y[1]

# Linear solution near the steady state
# c2 = (1/2)*(3*y0[0] + y0[1])
# c1 = y0[0] - c2
# x1_t = c1*np.exp(-t) + c2*np.exp(-3*t)
# x2_t = -3*c1*np.exp(-t) - c2*np.exp(-3*t)

# Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(t, current, label=r'Current ($x_1$)', color='blue')
# # plt.plot(t, x1_t, label=r'Linear sol ($x_1$)', color='k')
# plt.plot(t, voltage, label=r'Voltage ($x_2$)', color='red', linestyle='--')
# # plt.plot(t, x2_t, label=r'Linear sol ($x_2$)', color='k', linestyle='--')
# plt.title('RLC circuit simulation')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$x_1(t), x_2(t)$')
# plt.legend()
# plt.grid(True)
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(current, voltage, color='k')
plt.title('Trajectory in the phase space of a RLC circuit')
plt.xlabel(r'$x_1(t)$ (A)')
plt.ylabel(r'$x_2(t)$ (V)')
plt.legend()
plt.grid(True)
plt.show()








