"""Compute poles and zeros.

MECH 513, J.R. Forbes, 2025/09/05
Sample/starter code for students.
"""
# %%
# Import packages
import numpy as np
import control
from scipy import linalg

# %%
# Functions

def tm_zeros(P):
    """Zeros of square system.
    Input: square transfer matrix. 
    """

    # State-space form.
    P_ss = control.tf2ss(P)

    # Dimensions
    n_x = (P_ss.A).shape[0]
    n_u = (P_ss.B).shape[1]
    n_y = (P_ss.C).shape[0]
    M = np.block([[P_ss.A, P_ss.B], [P_ss.C, P_ss.D]])
    N = np.block([[np.eye(n_x), np.zeros((n_x, n_u))], [np.zeros((n_y, n_x)), np.eye(n_y)]])
    # Solve for zeros
    zeros,_ = scipy.linalg.eig(M,N)

    return zeros

def tm_poles(P):
    """Poles of system.
    Input: transfer matrix. 
    """

    # State-space form
    P_ss = control.tf2ss(P)

    # Solve for poles
    poles,_ = np.linalg.eig(P_ss.A)

    return poles

# %%
# Numerator and denominator

# G1
numerator_G1 = [[[1], [1]],
                [[2], [1]]]
denominator_G1 = [[[1, 1], [1, 2]],
                  [[1, 2], [1, 1]]]

# G2
numerator_G2 = [[[1, 2], [-1, -1]],
                [[0], [1]]]
denominator_G2 = [[[1, -np.sqrt(2)], [1, -np.sqrt(2)]],
                  [[1], [1]]]

# Transfer matrices
G1 = control.minreal(control.tf(numerator_G1, denominator_G1))
G2 = control.minreal(control.tf(numerator_G2, denominator_G2))
G3 = control.minreal(control.series(G1, G2))

# %%
# G1
G1_poles = tm_poles(G1)
print("The poles are: \n\n", G1_poles, "\n")

G1_zeros = tm_zeros(G1)
print("The zeros are: \n\n", G1_zeros, "\n")


# %%
# G2
G2_poles = tm_poles(G2)
print("The poles are: \n\n", G2_poles, "\n")

G2_zeros = tm_zeros(G2)
print("The zeros are: \n\n", G2_zeros, "\n")


# %%
# G3
G3_poles = tm_poles(G3)
print("The poles are: \n\n", G3_poles, "\n")

G3_zeros = tm_zeros(G3)
print("The zeros are: \n\n", G3_zeros, "\n")



# %%
