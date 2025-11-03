#%%
#Imports
import numpy as np
import matplotlib.pyplot as plt
import control
import scipy.signal as signal
#%%
#Initial Response
A = np.array([[0,1,0],
              [0,-.875,-20],
              [0,0,-50]])
B = np.array([[0],
              [0],
              [50]])
C = np.array([1,0,0])
D = np.array([0])

#%%
#check observability
Q0 = control.obsv(A,C)
print(Q0)
if np.linalg.matrix_rank(Q0) == Q0.shape[0]:
    print("Q0 is full rank: the system is observable.")
else:
    print("Q0 is not full rank: the system is not observable.")

#%%
#Place Observer Eigenvalues
p = [-12,-8+1j,-8-1j]
L  = control.place(A.T,C.T,p)
print(L)

#%%
#Combined dynamics

A_zeta = np.block([
    [A,                np.zeros((3,3))],
    [L.T.reshape(-1,1) @ C.reshape(1,-1), A - L.T.reshape(-1,1) @ C.reshape(1,-1)]
])
B_zeta = np.block([[B],
                  [B]])

C_zeta = np.identity(6)

D_zeta = np.zeros((6,1))
t = np.linspace(0,10,1000)
u = 0.025*signal.square(2*np.pi/2/2*t)

P_zeta = control.ss(A_zeta,B_zeta,C_zeta,D_zeta)


Tout_zeta, Yout_zeta = control.forced_response(P_zeta, t, u)
print(np.shape(Yout_zeta))
error = Yout_zeta[0:3,:] - Yout_zeta[3:6,:]
print(np.shape(error))

#plot on 3x1 plot deg, deg/s, N/m and plot the error x-x_hat
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(Tout_zeta, Yout_zeta[0,:])
axes[0].set_ylabel('theta (deg)')
axes[0].grid(True)
axes[1].plot(Tout_zeta, Yout_zeta[1,:])
axes[1].set_ylabel('omega (deg/s)')
axes[1].grid(True)
axes[2].plot(Tout_zeta, Yout_zeta[2,:])
axes[2].set_ylabel('torque (N·m)')
axes[2].set_xlabel('time (s)')
axes[2].grid(True)

fig.suptitle('forced response')
plt.tight_layout()
plt.show()

#%%
#plot the error
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(Tout_zeta, error[0,:])
axes[0].set_ylabel('theta (deg)')
axes[0].grid(True)
axes[1].plot(Tout_zeta, error[1,:])
axes[1].set_ylabel('omega (deg/s)')
axes[1].grid(True)
axes[2].plot(Tout_zeta, error[2,:])
axes[2].set_ylabel('torque (N·m)')
axes[2].set_xlabel('time (s)')
axes[2].grid(True)
fig.suptitle('error')