import numpy as np
from numpy import linalg as LA



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

# Q = np.array([[10, 0], [0, 1]])  # State cost matrix
# R = np.array([[1]])  # Control cost matrix
# # # Pendulum example
# dt = 0.015
# l = 10
# g = 9.81
# theta = np.pi
# norm2_list = []
# max_val = -1
# for i in range(100):
#     # P = np.array([[5.42156267, 1.8713373], [1.8713373, 0.80592194]])
#     # K = np.array([[-13.81818248, -4.44151409]])
#     # P = np.array([[4.07070798, 1.35050928], [1.35050928, 0.60804891]])
#     # K = np.array([[-12.10018314, -3.99232037]])
#     # P = np.array([[2.61719978, 0.82286531], [0.82286531, 0.41871497]])
#     # K = np.array([[-9.68932421, -3.17352989]])
#     # rho=0.95
#     # P = np.array([[51.15935795, 11.49689237], [11.49689237, 3.34257094]])
#     # K = np.array([[-29.26571774, -10.56473448]])
#     P = np.array([[34.92096505,  7.76312014], [ 7.76312014,  1.98764905]])
#     K = np.array([[-23.66438105,  -7.58198189]])
#     # P = np.array([[56.35692934, 12.3262689], [12.3262689, 3.27775006]])
#     # K = np.array([[-32.82044444, -10.40084854]])
#     # P = np.array([[220.09074475, 25.18699789], [25.18699789, 5.311497]])
#     # K = np.array([[-30.01183042, -12.3192521]])
#     B = np.array([[0], [1]]) * dt
#     A = np.array([[1, dt], [-g * np.cos(theta * i / 50) * dt / l, 1]])
#     # P = solve_discrete_are(A, B, Q, R)  # Discrete Algebraic Riccati Equation
#     # K = -np.linalg.inv(R) @ B.T @ P
#     print(K, P)
#     J = A + B @ K
#     transformed_J = transform_matrix(J, P)
#     # transformed_J = np.linalg.sqrtm(J)
#     max_round = np.max(np.sum(transformed_J, axis=1))
#     if max_val < max_round:
#         max_val = max_round
#     norm2 = LA.norm(transformed_J, ord=2)
#     norm2_list.append(norm2)
# print(norm2_list)
# max_norm2 = max(norm2_list)
# print(max_norm2)
# print("Max", max_val)




# car dynamics example
dt = 0.06
lr = 1.738
lf = 1.105


def range_float(start, end, step):
    while start < end:
        yield round(start, 2)
        start += step


norm2_list = []
val_list = []
# for delta in [x / 20.0 for x in range(-6, 6, 1)]:
#     for theta in [x / 200.0 for x in range(-114, 114, 1)]:
#         for v in [x / 20.0 for x in range(-10, 150, 1)]:
Q = np.diag([2,18,0.07, 0.005])  # State cost matrix
R = np.diag([20,2])  # Control cost matrix
max_val = -1

P = np.array([[ 2.69521645e-01,  7.85215178e-04,  9.27952796e-03,  1.01813882e-01],
    [ 7.85215178e-04,  4.02013898e-02,  1.47296528e-02, -2.81225696e-03],
    [ 9.27952796e-03,  1.47296528e-02,  1.55043788e+00,  6.97690875e-03],
    [ 1.01813882e-01, -2.81225696e-03,  6.97690875e-03,  2.03983167e-01]])
K = np.array([[-0.00723415, -0.01182626, -0.71176473,  0.0043664 ],
    [-0.8554823,  0.00754101,  0.02094794, -0.78042933]])

# P = np.array([[ 3.90195837, -0.23480136, -0.03832671,  1.91941839],
#     [-0.23480136,  1.71362198,  0.54547967, -0.19870702],
#     [-0.03832671,  0.54547967,  3.78164374,  0.02728795],
#     [ 1.91941839, -0.19870702,  0.02728795,  1.64985074]])

# K = np.array([[-0.02294105, -0.19971841, -1.1760203, 0.00074243],
#  [-1.90459251,  0.17521636, -0.05854881, -2.4105597]])
# P = np.array([[ 4.94329496e+01,  5.83202059e-02,  8.35729688e-02,
#          3.47850970e+01],
#        [ 5.83202059e-02,  6.72107980e+01,  2.09313726e+01,
#          6.84382572e-01],
#        [ 8.35729688e-02,  2.09313726e+01,  5.26761783e+01,
#         -1.14535396e+00],
#        [ 3.47850970e+01,  6.84382572e-01, -1.14535396e+00,
#          4.94589426e+01]])
# K = np.array([[ 2.44648768e-03, -2.15369621e+00, -1.34446230e+00,
#          1.60754041e-03],
#        [-1.04355291e+00, -2.05314772e-02,  3.43606188e-02,
#         -1.48376828e+00]])

# P = np.array([[ 46.16970391, -21.75544508,   0.85838513,  34.56394891],
#        [-21.75544508,  85.10084444,  24.69368248, -19.5893542 ],
#        [  0.85838513,  24.69368248,  57.98450617,  -0.90459615],
#        [ 34.56394891, -19.5893542 ,  -0.90459615,  52.43724884]]) 
# K = np.array([[-0.06101255, -2.03225782, -1.38745724,  0.05143319],
#        [-1.03691847,  0.58768063,  0.02713788, -1.57311747]])

for delta in range_float(-0.6, 0.6, 0.1):
    for theta in range_float(-0.8, 0.8, 0.1):
        for v in range_float(5, 12, 0.2):  # Lipschitz constant is sensitive to v
            beta_in = (lr * np.tan(delta)) / (lf + lr)
            beta = np.arctan(beta_in)

            term = ((lr / (np.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)

            A = np.array(
                [
                    [
                        1,
                        0,
                        -v * np.sin(theta + beta) * dt,
                        np.cos(theta + beta) * dt,
                    ],
                    [
                        0,
                        1,
                        v * np.cos(theta + beta) * dt,
                        np.sin(theta + beta) * dt
                    ],
                    [
                        0,
                        0,
                        1,
                        np.sin(beta) * dt / lr,

                    ],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array([[-v * np.sin(theta + beta) * dt * term, 0], 
                          [v * np.cos(theta + beta) * dt * term, 0],
                          [v * np.cos(beta) * dt * term / lr,0],
                          [0, dt]])
            # P = solve_discrete_are(A, B, Q, R)  # Discrete Algebraic Riccati Equation
            # K = -np.linalg.inv(R) @ B.T @ P # 2x4
            # # print(K, P)
            J = A + B @ K
            transformed_J = transform_matrix(J, P)
            norm2 = LA.norm(transformed_J, ord=2)
            norm2_list.append(norm2)
            val_list.append([delta, theta, v, P, K])
            max_round = np.max(np.sum(transformed_J, axis=1))
            if max_val < max_round:
                max_val = max_round
print(norm2_list)
max_norm2 = max(norm2_list)
print("values at max",val_list[np.argmax(norm2_list)])
print(max_norm2)
print("Max", max_val)



# values at max [-0.6, 0.4, 12.0, , ]

# values at max [0.5, -0.8, 12.0, ]

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
