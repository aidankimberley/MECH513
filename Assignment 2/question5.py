#Question 5

#%%
#libraries
import numpy as np
import control
from matplotlib import pyplot as plt

#%%
#define system

A = np.array([[-1,-1/10],
             [1,0]])
B = np.array([[0],
              [0]])
C = np.array([[1,0],
              [0,1]])
D = np.array([[0],
              [0]])
x0 = np.array([20,-5])

t = np.linspace(0,40,1000)

sys = control.ss(A,B,C,D)



T,y_out = control.initial_response(sys,t,x0)
#%%
#plot y_out
plt.figure(figsize=(10,5))
plt.plot(t,y_out[0,:],label='x1')
plt.plot(t,y_out[1,:],label='x2')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X output')
plt.title('X output')
plt.tight_layout()
plt.show()

#%%
#Find slope of modes over time
#mode 1 = v1
#mode 2 = v2
v1 = lambda x: (-5+5*np.sqrt(3/5))*x
v2 = lambda x: (-5-5*np.sqrt(3/5))*x

x1 = np.linspace(-2,20,1000)
x2_v1 = np.zeros(len(x1))
x2_v2 = np.zeros(len(x1))
for i in range(len(x1)):
    x2_v1[i] = v1(x1[i])
    x2_v2[i] = v2(x1[i])

#plot x1 vs x2 for both modes
#fix plot range to -30 and 20 in x2
plt.figure(figsize=(10,5))
plt.plot(x1,x2_v1,label='v1')
plt.plot(x1,x2_v2,label='v2')
plt.ylim(-30,20)
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Eigenvectors')
plt.tight_layout()
plt.show()

#%%
#different initial conditions
v1 = np.array([1,-5+5*np.sqrt(3/5)])
v2 = np.array([1,-5-5*np.sqrt(3/5)])

x0 = 20*v1
T,y_out = control.initial_response(sys,t,x0)
#plot y_out
plt.figure(figsize=(10,5))
plt.plot(t,y_out[0,:],label='x1')
plt.plot(t,y_out[1,:],label='x2')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X output')
plt.title('X: 20*v1')
plt.tight_layout()
plt.show()

#%%
x0 = 10*v1
T,y_out = control.initial_response(sys,t,x0)
#plot y_out
plt.figure(figsize=(10,5))
plt.plot(t,y_out[0,:],label='x1')
plt.plot(t,y_out[1,:],label='x2')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X output')
plt.title('X: 10*v1')
plt.tight_layout()
plt.show()

#%%
x0 = np.array([20,-5])
T,y_out = control.initial_response(sys,t,x0)
#plot y_out
plt.figure(figsize=(10,5))
plt.plot(t,y_out[0,:],label='x1')
plt.plot(t,y_out[1,:],label='x2')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X output')
plt.title('X: [20,-5]')
plt.tight_layout()
plt.show()