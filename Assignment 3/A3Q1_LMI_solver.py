# %%
# Libraries
import cvxpy
import numpy as np
import control
from matplotlib import pyplot as plt

a0h = 3
a0l = -5
a1h = 5
a1l = 1
a2h = -1.5
a2l = -2.5
#A is n x n
n=3
#B is n x m
m=1
def build_A(a0,a1,a2):
    return np.array([[0,1,0],
                     [0,0,1],
                     [-a0,-a1,-a2]])

A_1 = build_A(a0l,a1l,a2h)
A_2 = build_A(a0l,a1h,a2h)
A_3 = build_A(a0h,a1l,a2l)
A_4 = build_A(a0h,a1h,a2l)
print("A_1 = ", A_1)
B = np.array([[0], [0], [1]])

K_nom = np.array([[3.4495, 2.2975, 5.6871]])
F = cvxpy.Variable((m,n))
X = cvxpy.Variable((n,n),symmetric=True)
print("F = ", F)
print("X = ", X)
#%%
#LMI solver
objective = cvxpy.Minimize(cvxpy.norm(F - K_nom @ X, 'fro'))
constraints = [
    X >> 0,
    (A_1 @ X) + (X @ A_1.T) - (B @ F) - (F.T @ B.T) << 0,
    (A_2 @ X) + (X @ A_2.T) - (B @ F) - (F.T @ B.T) << 0,
    (A_3 @ X) + (X @ A_3.T) - (B @ F) - (F.T @ B.T) << 0,
    (A_4 @ X) + (X @ A_4.T) - (B @ F) - (F.T @ B.T) << 0,
]
prob = cvxpy.Problem(objective, constraints)
result = prob.solve(solver='MOSEK')
X_opt = X.value
F_opt = F.value
K_opt = np.linalg.solve(X_opt.T, F_opt.T)
K_opt = K_opt.T

print("Gain matrix K = ", K_opt)

#%%
#Get Closed Loop Initial Response
C = B.T
D = np.zeros((m,m))
a0nom = (a0h + a0l)/2
a1nom = (a1h + a1l)/2
a2nom = (a2h + a2l)/2
A_nom = build_A(a0nom,a1nom,a2nom)

A_cl_nom = A_nom - B @ K_opt
A_cl1 = A_1 - B @ K_opt
A_cl2 = A_2 - B @ K_opt
A_cl3 = A_3 - B @ K_opt
A_cl4 = A_4 - B @ K_opt

P_cl_nom = control.ss(A_cl_nom, B, C, D)
P_cl1 = control.ss(A_cl1, B, C, D)
P_cl2 = control.ss(A_cl2, B, C, D)
P_cl3 = control.ss(A_cl3, B, C, D)
P_cl4 = control.ss(A_cl4, B, C, D)

t = np.linspace(0,10,1000)
x0 = np.array([20/180*np.pi,15/180*np.pi,0.1])


Tout, Yout_nom = control.initial_response(P_cl_nom,t,x0)
Tout1, Yout1 = control.initial_response(P_cl1,t,x0)
Tout2, Yout2 = control.initial_response(P_cl2,t,x0)
Tout3, Yout3 = control.initial_response(P_cl3,t,x0)
Tout4, Yout4 = control.initial_response(P_cl4,t,x0)

fig, ax = plt.subplots()
ax.plot(Tout, Yout_nom, label='Nominal', color='C0', linestyle='-')
ax.plot(Tout1, Yout1, label='A_1', color='C1', linestyle='--')
ax.plot(Tout2, Yout2, label='A_2', color='C2', linestyle='-.')
ax.plot(Tout3, Yout3, label='A_3', color='C3', linestyle=':')
ax.plot(Tout4, Yout4, label='A_4', color='C4', linestyle=(0, (3, 1, 1, 1)))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Output')
ax.set_title('Initial Response Using K_opt as the gain matrix')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

#%%
#Now Using K_nom as the gain matrix
A_cl_nom = A_nom - B @ K_nom
A_cl1 = A_1 - B @ K_nom
A_cl2 = A_2 - B @ K_nom
A_cl3 = A_3 - B @ K_nom
A_cl4 = A_4 - B @ K_nom

P_cl_nom = control.ss(A_cl_nom, B, C, D)
P_cl1 = control.ss(A_cl1, B, C, D)
P_cl2 = control.ss(A_cl2, B, C, D)
P_cl3 = control.ss(A_cl3, B, C, D)
P_cl4 = control.ss(A_cl4, B, C, D)

Tout, Yout_nom = control.initial_response(P_cl_nom,t,x0)
Tout1, Yout1 = control.initial_response(P_cl1,t,x0)
Tout2, Yout2 = control.initial_response(P_cl2,t,x0)
Tout3, Yout3 = control.initial_response(P_cl3,t,x0)
Tout4, Yout4 = control.initial_response(P_cl4,t,x0)

fig, ax = plt.subplots()
ax.plot(Tout, Yout_nom, label='Nominal', color='C0', linestyle='-')
ax.plot(Tout1, Yout1, label='A_1', color='C1', linestyle='--')
ax.plot(Tout2, Yout2, label='A_2', color='C2', linestyle='-.')
ax.plot(Tout3, Yout3, label='A_3', color='C3', linestyle=':')
ax.plot(Tout4, Yout4, label='A_4', color='C4', linestyle=(0, (3, 1, 1, 1)))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Output')
ax.set_title('Initial Response Using K_nom as the gain matrix')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()