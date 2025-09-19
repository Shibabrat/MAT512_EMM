from sympy import symbols, diff, Matrix
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from pylab import rcParams
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
legend_fs = 16


# 1. Define the symbolic variable for derivative calculation
t, x1, x2 = symbols('t, x1, x2')
lambda1_vec = []
lambda2_vec = []

# 2. Define the parameters for evaluations
r1 = 0.10
r2 = 0.25
k1 = 10000
k2 = 6000
# a1 = r1 / k1, a2 = r2 / k2
# competition term, b1 = t * a1 and b2 = t * a2
# sensitivity analysis parameter, change this value to see the effect
t_val = 0.5
t_val_max = 0.6 # based on the given parameters for the competing plant species
num_t_val = 50

# 3. Define the RHS functions and coordinates of the co-existence steady state
denominator = (r1/k1)*(r2/k2) - t**2*(r1/k1)*(r2/k2)
x1star = (r1*(r2/k2) - r2*(t*(r1/k1)))/denominator
x2star = (r2*(r1/k1) - r1*(t*(r2/k2)))/denominator
f1 = r1*x1 - (r1/k1)*x1**2 - t*(r1/k1)*x1*x2
f2 = r2*x2 - (r2/k2)*x2**2 - t*(r2/k2)*x1*x2

# 4. Take the derivative with respect to alpha
df1dx1 = diff(f1, x1)
df1dx2 = diff(f1, x2)
df2dx1 = diff(f2, x1)
df2dx2 = diff(f2, x2)

df1dx1_val = diff(f1, x1).subs(t,t_val)
df1dx2_val = diff(f1, x2).subs(t,t_val)
df2dx1_val = diff(f2, x1).subs(t,t_val)
df2dx2_val = diff(f2, x2).subs(t,t_val)

# 5. Define the matrix of partial derivatives, A
A = Matrix([[df1dx1, df1dx2],[df2dx1, df2dx2]])

x1star_val = x1star.subs(t,t_val)
x2star_val = x2star.subs(t,t_val)
print('x*_1:',x1star_val, ',x*_2:',x2star_val)
lambda1 = list(A.eigenvals())[0].subs(t,t_val).subs(x1,x1star_val).subs(x2,x2star_val)
lambda2 = list(A.eigenvals())[1].subs(t,t_val).subs(x1,x1star_val).subs(x2,x2star_val)

t_val_vec = np.linspace(0, t_val_max, num_t_val)
for t_val in t_val_vec:
    x1star_val = x1star.subs(t,t_val)
    x2star_val = x2star.subs(t,t_val)
    lambda1 = list(A.eigenvals())[0].subs(t,t_val).subs(x1,x1star_val).subs(x2,x2star_val)
    lambda2 = list(A.eigenvals())[1].subs(t,t_val).subs(x1,x1star_val).subs(x2,x2star_val)
    lambda1_vec = np.append(lambda1_vec, lambda1)
    lambda2_vec = np.append(lambda2_vec, lambda2)


fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(t_val_vec, lambda1_vec, '-', color='k', ms=2.0, label=rf'$\lambda_1$')
ax.plot(t_val_vec, lambda2_vec, '-', color='k', ms=2.0, label=rf'$\lambda_2$')
ax.grid('on')
plt.show()

# eval_species1 = species1.subs(alpha, alpha_val)
# eval_species2 = species2.subs(alpha, alpha_val)
# eval_deriv1 = derivative1.subs(alpha,alpha_val)
# eval_deriv2 = derivative2.subs(alpha,alpha_val)
# eval_sensitivity1 = eval_deriv1*(alpha_val/eval_species1)
# eval_sensitivity2 = eval_deriv1*(alpha_val/eval_species2)

# # print({eval_deriv},{eval_senstivity})
# print("Sensitivity of the species 1 steady state population wrt alpha:",{eval_sensitivity1})
# print("Sensitivity of the species 2 steady state population wrt alpha:",{eval_sensitivity2})
 
 
 
 
 