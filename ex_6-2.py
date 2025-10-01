# Import libraries for python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define parameters for the model
r1 = 0.05
r2 = 0.08
k1 = 150000
k2 = 400000
alpha = 1*10**(-7)

# 2. Define the simulation parameters
x1_0 = 5000
x2_0 = 70000
y0 = [x1_0, x2_0]
t_final = 400
t_initial = 0
t_span = [t_initial, t_final] 
# Points in time where we want the solution
t_eval = np.linspace(t_span[0], t_span[1], 50)

# 3. Define the dynamic model
def whale_population(t, y, r1, r2, k1, k2, alpha):
    """
    Defines the system of first-order ODEs for the competing whale species model
    
    Args:
        t (float): Time (required by solve_ivp, but not used in this model).
        y (list or np.array): State vector [population of blue whale, x_1; population of fin whale, x_2].
        r1, r2 (float): intrinsic growth constants for blue and fin whales, respectively 
        k1, k2 (float): environmental capacity for blue and fin whales, respectively
        
    Returns:
        list: The derivatives [dx1/dt, dx2/dt].
    """
    x1, x2 = y  # Unpack the state vector
    dx1dt = r1*x1 - (r1/k1)*x1**2 - alpha*x1*x2 
    dx2dt = r2*x2 - (r2/k2)*x2**2 - alpha*x1*x2 
    
    return [dx1dt, dx2dt]

# 4. Run the simulation 
solution = solve_ivp(
    fun=whale_population,
    t_span=t_span,
    y0=y0,
    args=(r1, r2, k1, k2, alpha),
    t_eval=t_eval,
    dense_output=True
)

t = solution.t
x1_t = solution.y[0]
x2_t = solution.y[1]

# 5. Plotting
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
ax1.plot(t, x1_t, label=r'Blue whale ($x_1$)', color='m')
ax1.grid('on')
ax2 = fig.add_subplot(212)
ax2.plot(t, x2_t, label=r'Fin whale ($x_2$)', color='k', linestyle='--')
ax2.grid('on')
# plt.title('Competing whale species')
# plt.xlabel('Time (yrs)')
# plt.ylabel(r'$x_1(t), x_2(t)$')
plt.legend()
plt.grid(True)
plt.show()




 
