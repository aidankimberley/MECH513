#part 2

#%%
#Imports
import numpy as np
import control

#%%
#define system
G = control.tf([1,4,6],[1,10,11,44,66])
sys= control.tf2ss(G)
print(sys)

#%%
#check hand derived state space form
A = np.array([[0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [-66,-44,-11,-10]])
B = np.array([[0],
              [0],
              [0],
              [1]])
C = np.array([6,4,1,0])
D = np.array([0])

sys2 = control.ss(A,B,C,D)

tf2 = control.ss2tf(sys2)
print(tf2)
