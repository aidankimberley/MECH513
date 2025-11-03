#Assigment 1
#%%
#Imports
import numpy as np
import control
import matplotlib.pyplot as plt

#%%
#define system
A = np.array([
    [0, 1, 0, 0],
    [-10, -5, 10, 0],
    [0, 0, 0, 1],
    [5, 0, -5, -1/2]
])
B = np.array([[0,0],
             [1,0],
             [0,0],
             [0,1/2]])
C = np.array([[1,0,0,0],
             [0,0,1,0]])
D = np.array([[0,0],
             [0,0]])

sys = control.ss(A,B,C,D)
print(sys)
#%%
#intial response
q10 = 0.25
q1dot0 = 0
q20 = -0.5
q2dot0 = 0
#x defined as [q1, q2, q1dot, q2dot]
initial_state = np.array([q10, q20, q1dot0, q2dot0])
t = np.linspace(0, 20, 1000)
T, yout = control.initial_response(sys,t,initial_state=initial_state)
#get first row of yout
yout_q1 = yout[0,:]
yout_q2 = yout[1,:]

#%%
# Plot each output variable with labels
plt.figure()
for i in range(yout.shape[0]):
    plt.plot(T, yout[i], label=f'Output {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('System Initial Response')
plt.legend()
plt.grid(True)
plt.show()