from sympy import symbols, sin, diff

# 1. Define the symbolic variable
alpha = symbols('alpha')

r1 = 0.05
r2 = 0.08
k1 = 150000
k2 = 400000
# sensitivity analysis parameter, change this value to see the effect
alpha_val = 1*10**(-7)

equilibrium1 = k1*(alpha*r2*k2 - r1*r2)/(alpha**2*k1*k2 - r1*r2)
equilibrium2 = k2*(alpha*r1*k1 - r1*r2)/(alpha**2*k1*k2 - r1*r2)
# 2. Define the numerator and denominator functions
# numerator = 150000*(8000000*alpha - 1)
# denominator = (15*10**12*alpha**2 - 1)

# 3. Define the quotient function
# quotient_function = numerator / denominator

# 4. Take the derivative with respect to alpha
# derivative = diff(quotient_function, alpha)
derivative1 = diff(equilibrium1, alpha)
derivative2 = diff(equilibrium2, alpha)

# 5. Print the original function and its derivative
# print("Original Function: f(alpha) =", quotient_function)
# print("Derivative: f'(alpha) =", derivative)

eval_equilibrium1 = equilibrium1.subs(alpha, alpha_val)
eval_equilibrium2 = equilibrium2.subs(alpha, alpha_val)
eval_deriv1 = derivative1.subs(alpha,alpha_val)
eval_deriv2 = derivative2.subs(alpha,alpha_val)
eval_sensitivity1 = eval_deriv1*(alpha_val/eval_equilibrium1)
eval_sensitivity2 = eval_deriv1*(alpha_val/eval_equilibrium2)

# print({eval_deriv},{eval_senstivity})
print("Sensitivity of the equilibrium 1 wrt alpha:",{eval_sensitivity1})
print("Sensitivity of the equilibrium 2 wrt alpha:",{eval_sensitivity2})
 
 
 
 
 