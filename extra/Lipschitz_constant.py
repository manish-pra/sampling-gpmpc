import numpy as np
from numpy import linalg as LA


# # car dynamics example
# dt = 0.015
# lr = 1.738
# lf = 1.105


# def range_float(start, end, step):
#     while start < end:
#         yield round(start, 2)
#         start += step


# norm2_list = []
# # for delta in [x / 20.0 for x in range(-6, 6, 1)]:
# #     for theta in [x / 200.0 for x in range(-114, 114, 1)]:
# #         for v in [x / 20.0 for x in range(-10, 150, 1)]:
# for delta in range_float(-0.6, 0.6, 0.2):
#     for theta in range_float(-1.14, 1.14, 0.02):
#         for v in range_float(-1, 10, 0.2):  # Lipschitz constant is sensitive to v
#             beta_in = (lr * np.tan(delta)) / (lf + lr)
#             beta = np.arctan(beta_in)

#             term = ((lr / (np.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)

#             J = np.array(
#                 [
#                     [
#                         1,
#                         0,
#                         -v * np.sin(theta + beta) * dt,
#                         np.cos(theta + beta) * dt,
#                         -v * np.sin(theta + beta) * dt * term,
#                         0,
#                     ],
#                     [
#                         0,
#                         1,
#                         v * np.cos(theta + beta) * dt,
#                         np.sin(theta + beta) * dt,
#                         v * np.cos(theta + beta) * dt * term,
#                         0,
#                     ],
#                     [
#                         0,
#                         0,
#                         1,
#                         np.sin(beta) * dt / lr,
#                         v * np.cos(beta) * dt * term / lr,
#                         0,
#                     ],
#                     [0, 0, 0, 1, 0, dt],
#                 ]
#             )
#             norm2 = LA.norm(J, ord=2)
#             norm2_list.append(norm2)
# print(norm2_list)
# max_norm2 = max(norm2_list)
# print(max_norm2)

# Pendulum example
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
