"""Find transfer function from frequency response data.

J R Forbes, 2023/07/30.

Based on the SISO formulation found in [1] and [2].

References:
[1] E. S. Bañuelos-Cabral, J. A. Gutiérrez-Robles, and B. Gustavsen,
'Rational Fitting Techniques for the Modeling of Electric Power Components
and Systems Using MATLAB Environment', InTech, Dec. 06, 2017.

[2] A. O. Soysal and A. Semlyn, 'Practical Transfer Function Estimation and its
Application to Wide Frequency Range Representation of Transformers',
IEEE Trans. Power Delivery, Vol. 8, No. 3, July 1993.
"""

# %%
# Libraries
import numpy as np
import control
from scipy import linalg
from scipy import signal
from matplotlib import pyplot as plt


# %%
# Plotting parameters.
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Standard parameters.

# Laplace variable
s = control.tf('s')

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 11 / 2.54  # cm

# Convert rad / s to Hz
radpersec2Hz = 1 / 2 / np.pi
Hz2radpersec = 1 * 2 * np.pi


# %%
# Read data

data_read = np.loadtxt('student_freq_resp_data.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,)

f = data_read[:, 0].ravel()
omega = f * 2 * np.pi   # the frequencies in rad/s
G_real = data_read[:, 1].ravel()
G_imag = data_read[:, 2].ravel()

N_w = f.shape[0]
G = np.zeros(N_w, dtype=complex)
for i in range(N_w):
    G[i] = G_real[i] + 1j * G_imag[i]

# %%
# Compute freq response
N_f_max = f.shape[0]
G_mag = np.zeros(N_f_max,)
G_phase = np.zeros(N_f_max)
for i in range(N_f_max):
    G_mag[i] = np.sqrt((np.real(G[i]))**2 + (np.imag(G[i]))**2)  # absolute
    G_phase[i] = np.arctan2(np.imag(G[i]), np.real(G[i]))  # rad

G_mag_dB = 20 * np.log10(G_mag)
G_phase_deg = G_phase * 360 / 2 / np.pi
G_phase_deg = np.unwrap(G_phase_deg, period = 360)  # unwrap the phase

# Plot Bode plot
fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(f, G_mag_dB, '.', color='C3', label='System ID')
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
axes[1].semilogx(f, G_phase_deg, '.', color='C3', label='System ID')
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')
# fig.savefig(f'figs/system_ID_freq_resp.pdf')

# %%
# Custom LS problem for G(s) = b_1 s / (s^2 + a_1 s + a_0)

# Degree of numerator, degree of denominator.
m, n = 1, 2

# Number of numerator and denominator parameters.
N_num = m  # Note, in this case, there's just ONE numerator coefficient.
N_den = n

# Initialized A and b matrices that will be complex.
A = np.zeros((N_w, N_num + N_den), dtype=complex)
b = np.zeros((N_w, 1), dtype=complex)

N_skip = 10  # Don't necessarily need to use every data point.

# Nested for loop to populate A and b.
for i in range(0, N_w, N_skip):

    
    # ###### Fill in here, start ######
    # Populate b matrix.

    b[i] = -G[i]*omega[i]**2  # Placeholder, you must change.
    
    # Populate A matrix.
    coeff1 = omega[i]*1j #s
    coeff2 = -G[i] #Gi(s)
    coeff3 = -G[i]*omega[i]*1j #Gi(s)s
    A[i, :] = np.array([coeff1, coeff2, coeff3])
    # ###### Fill in here, end ######


# Split A and b into real and complex parts. Stack real part on top of
# complex part to create large A matrix.
A1 = np.block([[np.real(A)], [np.imag(A)]])
b1 = np.block([[np.real(b)], [np.imag(b)]])

# ###### Fill in here, start ######


# Solve Ax = b problem for x
x = linalg.lstsq(A1, b1)[0]

# Extract transfer function.
b1 = int(x[0])
a0 = int(x[1])
a1 = int(x[2])

G_est = control.tf([b1, 0], [1, a1, a0])  # Placeholder, you must change.
# print("Original transfer function: ", P, "\n")
print("Estimated transfer function: ", G_est)

# ###### Fill in here, end ######

# %%
# Bode plot of IDed tf

# Frequency range.
f_low = 0.01  # Hz
f_high = 1000  # Hz
N_w = np.int64(1e4)
w_shared = np.logspace(np.log10(f_low * Hz2radpersec), np.log10(f_high * Hz2radpersec), N_w)
f_shared_Hz = w_shared / 2 / np.pi
mag_est, phase_est, _ = control.frequency_response(G_est, w_shared)

# Convert to dB and deg.
mag_est_dB = 20 * np.log10(mag_est)
phase_est_deg = phase_est / np.pi * 180
# phase_est_deg = np.unwrap(phase_est, period = 360)  # unwrap the phase


# Bode plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
# fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(f, G_mag_dB, '.', color="C3", label='System ID')
ax[0].semilogx(f_shared_Hz, mag_est_dB, '-', color="C2", label='LS fit')
ax[0].set_yticks(np.arange(-80, 20, 20))
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[0].legend(loc='best')
# Phase plot
ax[1].semilogx(f, G_phase_deg, '.', color="C3")
ax[1].semilogx(f_shared_Hz, phase_est_deg, '-', color="C2")
ax[1].set_yticks(np.arange(-90, 210, 30))
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
# fig.savefig('figs/bode_fit.pdf')


# %%
# First principles model (normalized).
G_truth = control.TransferFunction([16.06, 0], [1, 20, 5678])  
mag_G_truth, phase_G_truth, _ = control.frequency_response(G_truth, w_shared)

# Convert to dB and deg.
mag_G_truth_dB = 20 * np.log10(mag_G_truth)
phase_G_truth_deg = phase_G_truth / np.pi * 180
# phase_est_deg = np.unwrap(phase_est, period = 360)  # unwrap the phase

# Bode plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
# fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(f_shared_Hz, mag_G_truth_dB, '-', color="C0", label='First Principles')
ax[0].semilogx(f, G_mag_dB, '.', color="C3", label='System ID')
ax[0].semilogx(f_shared_Hz, mag_est_dB, '-', color="C2", label='LS fit')
ax[0].set_yticks(np.arange(-80, 20, 20))
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[0].legend(loc='best')
# Phase plot
ax[1].semilogx(f_shared_Hz, phase_G_truth_deg, '-', color="C0")
ax[1].semilogx(f, G_phase_deg, '.', color="C3")
ax[1].semilogx(f_shared_Hz, phase_est_deg, '-', color="C2")
ax[1].set_yticks(np.arange(-90, 210, 30))
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
# fig.savefig('figs/bode_fit_with_truth.pdf')


# %%
# Relative error.

# Numerator relative error.
print("The numerator coefficient relative error is", np.abs(np.array([G_est.num]).ravel()[0] - np.array([G_truth.num]).ravel()[0]) / np.array([G_truth.num]).ravel()[0] * 100)

# Denominator relative error.
for i in range(1, 3, 1):
    print("The denominator coefficient relative error is", np.abs(np.array([G_est.den]).ravel()[i] - np.array([G_truth.den]).ravel()[i]) / np.array([G_truth.den]).ravel()[i] * 100)

# %%
