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

# 1. Define the analytical expression of the eigenvalues
lambda1 = lambda k: ((1 - 10*k) + np.sqrt((1 - 10*k)**2 - 20*k))/2
lambda2 = lambda k: ((1 - 10*k) - np.sqrt((1 - 10*k)**2 - 20*k))/2 
lambda1_vec = []
lambda2_vec = []

# 2. Evaluate the expressions for various k values, careful with complex values!
k_min = 0
k_max = 0.5
num_k = 1000
k_vec = np.linspace(k_min, k_max, num_k)
for i in range(num_k):
    if (1 - 10*k_vec[i])**2 - 20*k_vec[i] >= 0:
        lambda1_vec = np.append(lambda1_vec, lambda1(k_vec[i]))
        lambda2_vec = np.append(lambda2_vec, lambda2(k_vec[i]))
    else:
        real_comp = (1 - 10*k_vec[i])**2/4
        img_comp = (20*k_vec[i] - (1 - 10*k_vec[i])**2)/4
        lambda1_vec = np.append(lambda1_vec, real_comp + img_comp)
        lambda2_vec = np.append(lambda2_vec, real_comp + img_comp)

# 3. Plot the eigenvalues for the k-values
fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(k_vec, lambda1_vec, '-', color='k', ms=2.0, label=rf'$\lambda_1$')
ax.plot(k_vec, lambda2_vec, '-', color='m', ms=2.0, label=rf'$\lambda_2$')
ax.grid('on')
ax.legend(loc='lower left', fontsize=18, frameon=True)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\lambda$s')
plt.savefig(f'ex5-2_stability.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

 
 
 
 
 