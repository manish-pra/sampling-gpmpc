import numpy as np
from numpy import linalg as LA


# # car dynamics example
# dt = 0.06
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
# max_val = -1
# for delta in range_float(-0.6, 0.6, 0.1):
#     for theta in range_float(-0.8, 0.8, 0.02):
#         for v in range_float(5, 10, 0.2):  # Lipschitz constant is sensitive to v
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
#             max_round = np.max(np.sum(J, axis=1))
#             if max_val < max_round:
#                 max_val = max_round
# print(norm2_list)
# max_norm2 = max(norm2_list)
# print(max_norm2)
# print("Max", max_val)

# # # Pendulum example
# dt = 0.015
# l = 10
# g = 9.81
# theta = np.pi
# norm2_list = []
# max_val = -1
# for i in range(100):
#     J = np.array([[1, dt, 0], [-g * np.cos(theta * i / 50) * dt / l, 1, dt]])
#     max_round = np.max(np.sum(J, axis=1))
#     if max_val < max_round:
#         max_val = max_round
#     norm2 = LA.norm(J, ord=2)
#     norm2_list.append(norm2)
# print(norm2_list)
# max_norm2 = max(norm2_list)
# print(max_norm2)
# print("Max", max_val)


def transform_matrix(J, P):
    # Compute P^{1/2} using matrix square root (eigendecomposition)
    eigvals, eigvecs = np.linalg.eigh(P)  # P must be symmetric positive definite
    P_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    P_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    # Compute P^{-1/2} J P^{1/2}
    # transformed_J = P_sqrt @ J
    # transformed_J = P_inv_sqrt @ J @ P_sqrt
    transformed_J = P_sqrt @ J @ P_inv_sqrt
    # transformed_J = P_sqrt @ J.T @ P @ J @ P_inv_sqrt
    return transformed_J


from scipy.linalg import solve_discrete_are

Q = np.array([[10, 0], [0, 1]])  # State cost matrix
R = np.array([[1]])  # Control cost matrix
# # Pendulum example
dt = 0.015
l = 10
g = 9.81
theta = np.pi
norm2_list = []
max_val = -1
for i in range(100):
    # P = np.array([[5.42156267, 1.8713373], [1.8713373, 0.80592194]])
    # K = np.array([[-13.81818248, -4.44151409]])
    # P = np.array([[4.07070798, 1.35050928], [1.35050928, 0.60804891]])
    # K = np.array([[-12.10018314, -3.99232037]])
    # P = np.array([[2.61719978, 0.82286531], [0.82286531, 0.41871497]])
    # K = np.array([[-9.68932421, -3.17352989]])
    # rho=0.95
    # P = np.array([[51.15935795, 11.49689237], [11.49689237, 3.34257094]])
    # K = np.array([[-29.26571774, -10.56473448]])
    P = np.array([[34.92096505,  7.76312014], [ 7.76312014,  1.98764905]])
    K = np.array([[-23.66438105,  -7.58198189]])
    # P = np.array([[56.35692934, 12.3262689], [12.3262689, 3.27775006]])
    # K = np.array([[-32.82044444, -10.40084854]])
    # P = np.array([[220.09074475, 25.18699789], [25.18699789, 5.311497]])
    # K = np.array([[-30.01183042, -12.3192521]])
    B = np.array([[0], [1]]) * dt
    A = np.array([[1, dt], [-g * np.cos(theta * i / 50) * dt / l, 1]])
    # P = solve_discrete_are(A, B, Q, R)  # Discrete Algebraic Riccati Equation
    # K = -np.linalg.inv(R) @ B.T @ P
    print(K, P)
    J = A + B @ K
    transformed_J = transform_matrix(J, P)
    # transformed_J = np.linalg.sqrtm(J)
    max_round = np.max(np.sum(transformed_J, axis=1))
    if max_val < max_round:
        max_val = max_round
    norm2 = LA.norm(transformed_J, ord=2)
    norm2_list.append(norm2)
print(norm2_list)
max_norm2 = max(norm2_list)
print(max_norm2)
print("Max", max_val)
