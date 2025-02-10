import numpy as np
from numpy import linalg as LA

dt = 0.015
l = 10
g = 9.81
theta = np.pi
norm2_list = []
for i in range(100):
    J = np.array([[1, dt, 0], [-g * np.cos(theta * i / 50) * dt / l, 1, dt]])
    norm2 = LA.norm(J, ord=2)
    norm2_list.append(norm2)
print(norm2_list)
max_norm2 = max(norm2_list)
print(max_norm2)
