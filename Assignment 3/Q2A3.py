#Q2 A3
#%%
#Imports
import numpy as np
import matplotlib.pyplot as plt
import control
#%%
#Initial Response
A = np.array([[0,1,0],
              [0,-.875,-20],
              [0,0,-50]])
B = np.array([[0],
              [0],
              [50]])
C = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
D = np.array([[0],
              [0],
              [0]])
P = control.ss(A,B,C,D)
t = np.linspace(0,10,1000)
x0 = np.array([20/180*np.pi,15/180*np.pi,0.1])
Tout, Yout = control.initial_response(P,t,x0)
theta_deg = np.rad2deg(Yout[0, :])
omega_deg_s = np.rad2deg(Yout[1, :])

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(Tout, theta_deg)
axes[0].set_ylabel('theta (deg)')
axes[0].grid(True)

axes[1].plot(Tout, omega_deg_s)
axes[1].set_ylabel('omega (deg/s)')
axes[1].grid(True)

axes[2].plot(Tout, Yout[2, :])
axes[2].set_ylabel('torque (N·m)')
axes[2].set_xlabel('time (s)')
axes[2].grid(True)

fig.suptitle('Initial condition response')
plt.tight_layout()
plt.show()

#%%
#Check controllability
Qc = control.ctrb(A,B)
if np.linalg.matrix_rank(Qc) == Qc.shape[0]:
    print("Qc is full rank: the system is controllable.")
else:
    print("Qc is not full rank: the system is not controllable.")
#%%
#place eigenvalues
p = [-10,-2+1j,-2-1j]
K  = control.place(A,B,p)
print(K)

#%%
#closed loop initial response
A_cl = A - B @ K
P_cl = control.ss(A_cl, B, C, D)
Tout_cl, Yout_cl = control.initial_response(P_cl, t, x0)
theta_deg_cl = np.rad2deg(Yout_cl[0, :])
omega_deg_s_cl = np.rad2deg(Yout_cl[1, :])

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(Tout_cl, theta_deg_cl)
axes[0].set_ylabel('theta (deg)')
axes[0].grid(True)

axes[1].plot(Tout_cl, omega_deg_s_cl)
axes[1].set_ylabel('omega (deg/s)')
axes[1].grid(True)

axes[2].plot(Tout_cl, Yout_cl[2, :])
axes[2].set_ylabel('torque (N·m)')
axes[2].set_xlabel('time (s)')
axes[2].grid(True)

fig.suptitle('Closed-loop initial condition response')
plt.tight_layout()
plt.show()

