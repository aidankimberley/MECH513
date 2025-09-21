#Question 4

#%%
#Imports
import numpy as np
import control
import matplotlib.pyplot as plt

#%%
A = np.array([[0,-2],
             [1,-12]])
B = np.array([[1],
              [0]])
C = np.array([0,1])
D = np.array([0])

sys = control.ss(A,B,C,D)
G = control.ss2tf(sys)
print(G)

eigenvalues, eigenvectors = np.linalg.eig(A)

Gden = np.array(G.den).ravel()
roots = np.roots(Gden)
print("eigenvalues: ", eigenvalues)
print("roots of the denominator: ", roots)



# %%
#impulse response
t = np.linspace(0, 60, 1000)
T, yout= control.impulse_response(sys, t)
#plot

plt.plot(T, yout)
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Impulse Response')
plt.grid(True)
plt.show()


# %%
#step response
ICs = np.array([0.5, -0.25])
T, yout= control.step_response(sys, t, initial_state=ICs)
#plot
plt.plot(T, yout)
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Step Response')
plt.show()

# %%
#forced response
import scipy.signal as signal
u = 5 * signal.square(1/20*2*np.pi/2*t)
T, yout= control.forced_response(sys, t, u, initial_state=ICs)

# Create stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot input signal
ax1.plot(T, u, 'b-', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Input')
ax1.set_title('Input Signal (Square Wave)')
ax1.grid(True)

# Plot forced response
ax2.plot(T, yout, 'r-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Output')
ax2.set_title('Forced Response')
ax2.grid(True)

plt.tight_layout()
plt.show()