from sympy import symbols, sin, diff

# 1. Define the symbolic variable for derivative calculation
alpha = symbols('alpha')

# 2. Define the parameters for evaluations
r1 = 0.05
r2 = 0.08
k1 = 150000
k2 = 400000
# sensitivity analysis parameter, change this value to see the effect
alpha_val = 1*10**(-7)

# 3. Define the functions for the co-existence steady state
# numerator = 150000*(8000000*alpha - 1)
# denominator = (15*10**12*alpha**2 - 1)
# equilibrium1 = k1*(alpha*r2*k2 - r1*r2)/(alpha**2*k1*k2 - r1*r2)
# equilibrium2 = k2*(alpha*r1*k1 - r1*r2)/(alpha**2*k1*k2 - r1*r2)
numerator1 = k1*(alpha*(k2/r1) - 1)
numerator2 = k2*(alpha*(k1/r2) - 1)
denominator = (alpha**2*(k1/r1)*(k2/r2) - 1)
species1 = numerator1 / denominator
species2 = numerator2 / denominator

# 4. Take the derivative with respect to alpha
derivative1 = diff(species1, alpha)
derivative2 = diff(species2, alpha)

# 5. Print the original function and its derivative
# print("Original Function: f(alpha) =", equilibrium1)
# print("Original Function: f(alpha) =", equilibrium2)
# print("Derivative: f'(alpha) =", derivative1)
# print("Derivative: f'(alpha) =", derivative2)

eval_species1 = species1.subs(alpha, alpha_val)
eval_species2 = species2.subs(alpha, alpha_val)
eval_deriv1 = derivative1.subs(alpha,alpha_val)
eval_deriv2 = derivative2.subs(alpha,alpha_val)
eval_sensitivity1 = eval_deriv1*(alpha_val/eval_equilibrium1)
eval_sensitivity2 = eval_deriv1*(alpha_val/eval_equilibrium2)

# print({eval_deriv},{eval_senstivity})
print("Sensitivity of the species 1 steady state population wrt alpha:",{eval_sensitivity1})
print("Sensitivity of the species 2 steady state population wrt alpha:",{eval_sensitivity2})
 
 
 
 
 