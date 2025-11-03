"""Sample code for computing H2 and Hinf norms.

J R Forbes, 2023/03/11
"""

# %%
# Packages
import numpy as np
from scipy import linalg
import control
import cvxpy
from matplotlib import pyplot as plt


# %%
# Functions


def SVP(P, w, max_tf):
    """Singular value plot.

    P : Transfer Matrix, object.
    w : frequencies, real.
    max_tf : plot max singular value, True or False.

    Based on
    https://python-control.readthedocs.io/en/0.9.3.post2/robust_mimo.html
    """
    m, p, _ = P.frequency_response(w)  # mag, phase

    # Compute singular values
    sigma_jw = (m * np.exp(1j * p)).transpose(2, 0, 1)  # The transpose
    sigma = (np.linalg.svd(sigma_jw, compute_uv=False)).T

    if max_tf == True:
        sigma = (np.max(sigma, 0)).reshape((1, -1))

    sigma_dB = 20 * np.log10(sigma)  # Singular values in dB
    n_s = sigma_dB.shape[0]

    fig, ax = plt.subplots()
    # singular value plot
    ax.set_xlabel(r'$\omega$ (rad/s)')
    ax.set_ylabel(r'Magnitude (dB)')
    for i in range(n_s):
        ax.semilogx(w, sigma_dB[i, :])

    return sigma, w, plt


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Mass-spring_damper system.
m = 20  # kg, mass
d = 0.1  # N s / m, damper
k = 500  # N / m, spring
# Form state-space matrices.
A = np.array([[0, 1],
              [-k / m, -d / m]])
B = np.array([[0],
              [1 / m]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])
n_x, n_u, n_y = A.shape[0], B.shape[1], C.shape[0]
G = control.ss(A, B, C, D)
G_tf = control.tf(G)
print(f"G(s) = {G_tf}")


#Check stability of G(s)
poles = control.poles(G)
print(f"Poles of G(s): {poles}")


# %%
# Plot singular value plot.
w = np.logspace(-1, 3, 50000)
sigma, w, plt = SVP(G, w, True)
print(np.shape(sigma))
# Plot the peak value.
# To do so, find the peak value, and find at what frequency the
peak_index = np.argmax(sigma)
print(f"Peak index: {peak_index}")
w_peak = w[peak_index]
print(f"Peak frequency: {w_peak}")
print(f"Peak sigma: {sigma[0, peak_index]}")
print(f"Peak value on plot: {20*np.log10(sigma[0, peak_index])}")

# peak value occurs. You might find np.argmax helpful.
plt.plot(w_peak, 20*np.log10(sigma[0,peak_index]), '*', color='C3')  # You modify this.
plt.title("Maximum singular value of G(s)")
plt.show()
# plt.savefig('SVP.pdf')


# %%
# Compute Hinf norm via LMI.
# Use this as an example of how to use cvxpy to solve LMI problems.

# Create cvxpy variables.
P = cvxpy.Variable((n_x, n_x), symmetric=True)
gamma_P = cvxpy.Variable((1, 1), symmetric=True)

# Create objective.
# Minimization problem.
objective = cvxpy.Minimize(gamma_P)

# Create constraints.
F0 = cvxpy.bmat([[P @ A + A.T @ P, P @ B, C.T],
                 [(P @ B).T, -gamma_P * np.eye(n_u), D.T],
                 [C, D, -gamma_P * np.eye(n_y)]])

# Augment constraints.
constraints = [
    gamma_P >> 0,
    P >> 0,
    F0 << 0
]

# Create and solve problem.
prob = cvxpy.Problem(objective, constraints)
result = prob.solve(solver='MOSEK', verbose=False)

# Extract results.
gamma_P_LMI = (gamma_P.value).item()
print(f'The closed-loop Hinf norm of G(s) is {gamma_P_LMI}\n')

# %%
# Compute closed-loop H2 norm via Lyapunov equation.
# Zhou Doyle, Essentials of Robust Control, Lemma 4.4, pg 53.
# Compute the closed-loop H2 norm using the observability Gramian solution.
Q = control.lyap(A.T, C.T @ C)
nu_Q = np.sqrt(np.trace(B.T @ Q @ B))
print(f'The closed-loop H_2 norm of G(s) is {nu_Q}.\n')

# %%
# You code LMI form of the H2 norm computation here.
X = cvxpy.Variable((n_x, n_x), symmetric=True)
Z = cvxpy.Variable((n_u, n_u), symmetric=True)
nu_sq = cvxpy.Variable((1, 1))
constraints = [
    X>>0,
    Z>>0,
    cvxpy.trace(Z) << nu_sq
 ]
LMI1 = cvxpy.bmat([
    [A @ X + X @ A.T, X @ C.T],
    [C @ X, -np.eye(n_y)]
])

LMI2 = cvxpy.bmat([
    [Z, B.T],
    [B, X]
])
constraints.append(LMI1 << 0)
constraints.append(LMI2 >> 0)  # Note: >> 0 means positive semidefinite, << 0 means negative semidefinite
objective = cvxpy.Minimize(nu_sq)
prob = cvxpy.Problem(objective, constraints)
result = prob.solve(solver='MOSEK', verbose=False)

# Check if problem solved successfully
if prob.status not in ['optimal', 'optimal_inaccurate']:
    print(f'Warning: Problem status is {prob.status}')
    print(f'Problem value: {prob.value}')

# Extract results.
if nu_sq.value is not None:
    nu_sq_LMI = (nu_sq.value).item()
    print(f'The closed-loop H_2 norm of G(s) (via LMI) is {np.sqrt(nu_sq_LMI)}.\n')
else:
    print('Error: Problem did not solve successfully. nu_sq.value is None.')

# %%
