"""SVD-based control of Distillation Process (LV Configuration)

J R Forbes, 2025/01/08

Sample code for students.

Based on Skogestad Section 3.5.4.
"""

# %%
# Packages
import numpy as np
from scipy import signal
import control
from matplotlib import pyplot as plt

# %%
# Functions


def SVP(P, w, max_tf):
    """Singular value plot.

    P : Transfer Matrix, object.
    w : frequncies, real.
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
# Common parameters

# Time
t_start, t_end, dt = 0, 40, 1e-2
t = np.arange(t_start, t_end, dt)
n_t = t.shape[0]

# Laplace variable s
s = control.tf('s')

# Frequencies
omega_count = 100
omega = np.logspace(-2, 2, omega_count)

# %%
# Command and noise requirements. 
omega_r_high = 0.1  # rad / s
omega_r = np.extract(omega <= omega_r_high, omega)
omega_n_low = 10  # rad / s
omega_n = np.extract(omega >= omega_n_low, omega)

gamma_r = 10**(-5 / 20)
gamma_n = 10**(-20 / 20)

gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(omega_r.shape[0],)
gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(omega_n.shape[0],)

fig, ax = plt.subplots()
# Magnitude plot
ax.semilogx(omega_r, gamma_r_dB, '-', color='C3', label=r'$\gamma_r$')
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
ax.set_xlim([omega[0], omega[-1]])
ax.set_ylim([-40, 20])
ax.legend(loc='lower right')
# plt.savefig('figs/command_spec.pdf')

fig, ax = plt.subplots()
# Magnitude plot
ax.semilogx(omega_n, gamma_n_dB, '-', color='C6', label=r'$\gamma_n$')
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
ax.set_xlim([omega[0], omega[-1]])
ax.set_ylim([-40, 20])
ax.legend(loc='lower left')
# plt.savefig('figs/noise_spec.pdf')


# %%
# MIMO system, Distillation Process, LV configuration.
num = [[[87.8], [-86.4]],
       [[108.2], [-109.6]]]
den = [[[75, 1], [75, 1]],
       [[75, 1], [75, 1]]]
P = control.tf(num, den)
n_y, n_u, _ = np.array(P.num, dtype=object).shape
n_s = np.min([n_y, n_u])

# Plot singular values.
sigma_P, _, plt = SVP(P, omega, False)
# plt.savefig('figs/P_vs_omega.pdf')


# %%
# SVD-based control.

# SVD of constant part of the plant. 
P_const = np.array(num).reshape((2, 2))

# You will need this SVD.
U, S, V_herm = np.linalg.svd(P_const, full_matrices=True)

# ####### MODIFY START #######

# Controller.
# YOU MUST MODIFY.

# Implement the SVD-based controller from the handout
# K(s) = V (k(s) Σ^{-1}) U^T,
# where Σ has singular values σ1, σ2 from the SVD of the constant plant P0,
# and k(s) = k1 / (P0(s) s). For this process P0(s) = 1/(75 s + 1),
# so k(s) = k1 (75 s + 1) / s.
k_1 = 1.5
# Create the scalar compensator k(s) used for both channels
k_s = k_1 * (75 * s + 1) / s
# Form diag(k(s)/σ1, k(s)/σ2). The array S holds [σ1, σ2].
# We create a block-diagonal LTI system using control.append.
K_diag = control.append(k_s / S[0], k_s / S[1])

# Convert the unitary matrices V and U^T into static (gain-only) LTI systems
# so they can be multiplied with transfer functions. This yields
# K(s) = V · diag(k/σ) · U^T.
V = V_herm.conj().T
K = control.ss([], [], [], V) * K_diag * control.ss([], [], [], U.T)





# ####### MODIFY END #######

# SV plot of controller. 
sigma_K, _, plt = SVP(K, omega, False)
# plt.savefig('figs/sigma_K_vs_omega.pdf')


# %%
# L_O
L_O = control.minreal(P * K)

sigma_L_O, _, plt = SVP(L_O, omega, False)
sigma_L_O_dB = 20 * np.log10(sigma_L_O)


# %%
# S_0
I = control.ss([], [], [], np.eye(n_y))
S_O = I.feedback(P * K, -1)  # negative feedback
eig_A_S_O = np.linalg.eigvals(S_O.A)
print("The eigenvalues of the A matrix are", eig_A_S_O)

sigma_S_O, _, plt = SVP(S_O, omega, False)
sigma_S_O_dB = 20 * np.log10(sigma_S_O)

# Plot again, with the command following specification. 
fig, ax = plt.subplots()
# Magnitude plot
ax.semilogx(omega_r, gamma_r_dB, '-', color='C3', label=r'$\gamma_r$')
ax.semilogx(omega, sigma_S_O_dB[0, :], '-', label=r'$\sigma_{\max}(\mathbf{S}_{\mathrm{O}}(j \omega))$')
ax.semilogx(omega, sigma_S_O_dB[1, :], '-', label=r'$\sigma_{\min}(\mathbf{S}_{\mathrm{O}}(j \omega))$')
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower right')
# plt.savefig('figs/sigma_S_O_vs_omega.pdf')


# %%
# T_O
T_O = control.minreal(L_O * S_O)

sigma_T_O, _, plt = SVP(T_O, omega, False)
sigma_T_O_dB = 20 * np.log10(sigma_T_O)

# Plot again, with the noise mitigation specification. 
fig, ax = plt.subplots()
# Magnitude plot
ax.semilogx(omega_n, gamma_n_dB, '-', color='C6', label=r'$\gamma_n$')
ax.semilogx(omega, sigma_T_O_dB[0, :], '-', label=r'$\sigma_{\max}(\mathbf{T}_{\mathrm{O}}(j \omega))$')
ax.semilogx(omega, sigma_T_O_dB[1, :], '-', label=r'$\sigma_{\min}(\mathbf{T}_{\mathrm{O}}(j \omega))$')
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
# plt.savefig('figs/sigma_T_O_vs_omega.pdf')


# %%
# Command (reference)
_, command = control.step_response(control.tf([1], [1 / (omega_r_high), 1]), t)
r = np.block([[command], [-command]])

# Noise
np.random.seed(123321)
mu, sigma = 0, gamma_n / 3  # 3 sigma
noise_raw = np.random.normal(mu, sigma, (2, n_t))
# Butterworth filter, high pass
b_bf, a_bf = signal.butter(6, omega_n_low, 'high', analog=True)
G_bf = control.tf(b_bf, a_bf)
_, noise = control.forced_response(control.minreal(control.append(G_bf, G_bf)), t, noise_raw)

# Response
_, y = control.forced_response(T_O, t, r - noise)
_, e = control.forced_response(S_O, t, r - noise)

# Plot response, y and e. 
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$y(t)$ (units)')
ax[1].set_ylabel(r'$e(t)$ (units)')
# Plot data
ax[0].plot(t, r[0, :], '--', color='C3', label=r'$r_1$')
ax[0].plot(t, r[1, :], '--', color='C2', label=r'$r_2$')
ax[0].plot(t, y[0, :], '-', color='C0', label=r'$y_1$')
ax[0].plot(t, y[1, :], '-', color='C1', label=r'$y_2$')
ax[1].plot(t, e[0, :], '-', color='C0', label=r'$e_1$')
ax[1].plot(t, e[1, :], '-', color='C1', label=r'$e_2$')
ax[1].set_ylim([-1, 1])
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='best')
fig.tight_layout()
# fig.savefig('figs/y_e_system_response.pdf')
# plt.show()


# %%