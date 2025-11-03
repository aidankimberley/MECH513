"""
SISO system ID.

J R Forbes, 2025/01/22

Sample code for students. 
"""

# %%
# libraries 
import numpy as np
import control
from scipy import signal
from matplotlib import pyplot as plt
from scipy import fft


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Standard parameters.

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 11 / 2.54  # cm

# Convert rad / s to Hz
radpersec2Hz = 1 / 2 / np.pi
Hz2radpersec = 1 * 2 * np.pi


# %%
# Read data
data_read = np.loadtxt('io_data.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,)

t = data_read[:, 0].ravel()
y = data_read[:, 1].ravel()
u = data_read[:, 2].ravel()

# dt
dt = t[1] - t[0]


# %%
# Chirp frequency bounds
chirp_freq_low = 0.1  # Hz, min frequency
chirp_freq_high = 100  # Hz, max frequency


# %%
# FFT of the input and output
N = t.size

# Single-sided FFT of input
u_fft = fft.rfft(u, n=N) / N  # same units as u
u_mag = np.abs(u_fft)  # compute the magnitude of each u_fft
u_mag[1:] = 2 * u_mag[1:]  # multiply all mag's by 2, but the zero frequency
u_phase = np.angle(u_fft, deg=False)  # compute the angle

f = fft.rfftfreq(N, d=dt)  # the frequencies in Hz
N_f_max = np.searchsorted(f, chirp_freq_high)  # find index of max frequency
omega = f * 2 * np.pi   # the frequencies in rad/s
w = omega[:N_f_max]

# Recompute the FFT with the correct scaling
u_FFT = np.zeros(N_f_max, dtype=complex)
for i in range(N_f_max):
    u_FFT[i] = u_mag[i] * np.cos(u_phase[i]) + 1j * u_mag[i] * np.sin(u_phase[i])


# ###### Fill in here, start ######
# You must now compute y_FFT
y_fft = fft.rfft(y, n=N) / N
y_mag = np.abs(y_fft)
y_phase = np.angle(y_fft, deg=False)

y_FFT = np.zeros(N_f_max, dtype=complex)
for i in range(N_f_max):
    y_FFT[i] = y_mag[i] * np.cos(y_phase[i]) + 1j * y_mag[i] * np.sin(y_phase[i])



# ###### Fill in here, end ######

# %% compute TF
# N_omega = omega.size
G = np.zeros(N_f_max, dtype=complex)
G_mag = np.zeros(N_f_max,)
G_phase = np.zeros(N_f_max)
for i in range(N_f_max):
    # ###### Fill in here, start ######
    # You must compute G at each frequency
    G[i] = y_FFT[i] / u_FFT[i]

    # ###### Fill in here, end ######
    
    # Compute magnitude and phase of G.
    G_mag[i] = np.sqrt((np.real(G[i]))**2 + (np.imag(G[i]))**2)  # absolute
    G_phase[i] = np.arctan2(np.imag(G[i]), np.real(G[i]))  # rad

G_mag_dB = 20 * np.log10(G_mag)
G_phase_deg = G_phase * 360 / 2 / np.pi
# G_phase_deg = np.unwrap(G_phase_deg, period = 360)  # unwrap the phase


# %%
# Plots

# Find index of chirp_freq_low
f_differences = np.abs(f - chirp_freq_low)
N_f_min = f_differences.argmin()

#plot time domain input and output
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.plot(t, u, label='Input')
ax.plot(t, y, label='Output')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend(loc='best')
fig.suptitle('Time Domain Input and Output')
fig.tight_layout()
plt.show()




#%%
# Plot FFT of input and output
# Plot FFT of input
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(u_mag[N_f_min:N_f_max]), '.', color='C0', label=r'$|u(j\omega)|$')
# ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(y_mag[N_f_min:N_f_max]), '.', color='C1', label=r'$|y(j\omega)|$')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
fig.suptitle('FFT of Input')
fig.tight_layout()
plt.show()
#plot FFT of output with title "FFT of Output"
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(y_mag[N_f_min:N_f_max]), '.', color='C1', label=r'$|y(j\omega)|$')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
fig.suptitle('FFT of Output')
fig.tight_layout()
plt.show()
# fig.savefig(f'figs/IO_freq_resp.pdf')

# Plot Bode plot
fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(f[N_f_min:N_f_max], G_mag_dB[N_f_min:N_f_max], '.', color='C3', label='System ID')
axes[0].set_yticks(np.arange(-80, 20, 20))
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
# axes[1].semilogx(f_shared_Hz, phase_G_deg, color='C2', label='True')
axes[1].semilogx(f[N_f_min:N_f_max], G_phase_deg[N_f_min:N_f_max], '.', color='C3', label='System ID')
axes[1].set_yticks(np.arange(-90, 210, 30))
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')
# fig.savefig(f'figs/system_ID_freq_resp.pdf')


# %%
# Save data

data_write = np.block([[f[N_f_min:N_f_max]], [np.real(G[N_f_min:N_f_max])], [np.imag(G[N_f_min:N_f_max])]]).T
np.savetxt('student_freq_resp_data.csv',
        data_write,
        fmt='%.8f',
        delimiter=',',
        header='f (Hz), Re{G(jw)}, Im{G(jw)}')


# %%
# Plot show
plt.show()

# %%
