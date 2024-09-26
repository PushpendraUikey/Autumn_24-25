
"""
23b1023: Pushpendra Uikey
23b1024: Nischal
23b0993: Nithin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import binom, norm
from scipy.stats import gamma

###################### Task-A ##########################

# load the data
data = np.loadtxt('3.data')

# Compute the 1st Moment (mean)
first_moment = np.mean(data)

# Compute the 2nd moment (variance-like moment)
second_moment = np.mean(data ** 2)

print("First Moment (Mean):", first_moment)
print("Second Moment:", second_moment)


#################### Task-B ###################

plt.hist(data, bins=100, edgecolor=None, alpha=0.7)
plt.title('Histogram of Data')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("3b.png")


#################### Task-C ####################

def moment_for_Bin(vars):
    n, p = vars
    m1 = n * p - first_moment
    m2 = n * p * (1-p) + (n*p)**2 - second_moment  # E[X^2] = var(x) + (E[x])^2

    return [m1, m2]


initial_guess = [20, 0.3]  # initial guess for n and p;

# Use fsolve to solve the system of equations
n, p = fsolve(moment_for_Bin, initial_guess)

n = round(n)

# Print the results
print(f"Estimated n: {n}")
print(f"Estimated p: {p}")


k = np.arange(0, n+1)

# Compute the PMF for each k value
pmf_values = binom.pmf(k, n, p)

# Plotting
plt.figure(figsize=(10, 6))

# histogram
plt.hist(data, bins=100, edgecolor=None, alpha=0.7, density=True, label='Histogram')

plt.plot(k, pmf_values, 'r-', label='Binomial PMF') 

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data with Binomial PMF')
plt.legend()
# plt.grid(True)
plt.savefig("3c.png")


#####################- Task-D #############################

## nth G_moment = theta^n * Gamma(n+k)/Gamma(k)

# for scipy Fsolve function
def first_two_gammaMoment(var):
    k, theta = var
    g1 = theta * k - first_moment
    g2 = (theta**2)*k*(k+1) - second_moment

    return [g1, g2]

#initial_guess of k and theta
initial_guess = [ 9.6, 0.6 ]

# use of fsolve to solve the system of equation

k, theta = fsolve(first_two_gammaMoment, initial_guess)

print(f"Estimated K: {k}")
print(f"Estimated theta: {theta}")


# Define the range of x values
# x = np.arange(0, n+1)
x = np.linspace(0,n,100)

# Compute the Gamma PDF
pdf_values = gamma.pdf(x, a=k, loc=0, scale=theta)

plt.figure(figsize=(10, 6))

# histogram
plt.hist(data, bins=100, edgecolor=None, alpha=0.7, density=True, label='Histogram')

# Plot the Gamma PDF
plt.plot(x, pdf_values, 'r-', label=f'Gamma PDF (k={k:.3f}, θ={theta:.3f})')
plt.title('Gamma Distribution PDF')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('3d.png')


######################- Task-E ######################

# log-likelihood of Bin(n,p)(x)

Bin_prob = binom.pmf(data.round(),n,p)
log_like_Bin = np.log(Bin_prob + 1e-10).sum() / data.size     # adding small number to avoid log(0)

# log-likelihood of Gamma(n,p)(x)

gamma_prob = gamma.pdf(data, a = k, scale = theta)
log_like_Gamma = np.log(gamma_prob + 1e-10).sum() / data.size

print(f"likelihood is : { log_like_Bin } ")
print(f"likelihood of gamma distribution is: {log_like_Gamma}")

print(f"Better fit distribution is {"Binomial Distribution!" if log_like_Bin > log_like_Gamma else "Gamma Distribution!"}")

####################- Task-F ##################

## third moment generator
third_moment = np.mean(data ** 3)

# fourth moment genenrator
fourth_moment = np.mean(data**4)

def gmm_Moment(var):
    u1, p1, u2, p2 = var

    gmm1 = p1*u1 + p2*u2    - first_moment
    gmm2 = p1 * (1+(u1**2)) + p2 *( 1 + (u2**2))  -  second_moment
    gmm3 = p1*((u1**3) + 3*u1) + p2*((u2**3) + 3*u2)  - third_moment
    gmm4 = p1*((u1**4) + 6*(u1**2) + 3) + p2*((u2**4) + 6*(u2**2) + 3)   - fourth_moment

    return [gmm1, gmm2, gmm3, gmm4]

# initial guesses
initial_guess = [5, 0.6 , 9, 0.4]

# use of fsolve to solve the problem
u1, p1, u2, p2 = fsolve(gmm_Moment, initial_guess)
 
# print(f"Solution: {u1} {p1} {u2} {p2}")
# using lispace to generate value's in equal range
x = np.linspace(0,n+1,100)

# gmm pdf 
gmm_pdf = p1 * norm.pdf(x, loc=u1, scale=1) + p2*norm.pdf(x, loc=u2, scale=1) #scale is 1 as std 1 given

# histogram
plt.hist(data, bins=100, edgecolor=None, alpha=0.7, density=True, label='Histogram')

plt.plot(x, gmm_pdf, 'r-', label="GMM")

plt.title("GMM fit to data")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('3f.png')
