import torch
import gpytorch
import numpy as np
from casadi import SX, MX, vertcat
import casadi as cas


def generate_gp_funs(gp_model, covar_jac=False, B=None):
    if gp_model.train_inputs[0].device.type == "cuda":
        to_tensor = lambda X: torch.Tensor(X).cuda()
        to_numpy = lambda T: T.cpu().numpy()
    else:
        to_tensor = lambda X: torch.Tensor(X)
        to_numpy = lambda T: T.numpy()

    if B is not None:
        B_tensor = to_tensor(B)

    def mean_fun_sum(y):
        with gpytorch.settings.fast_pred_var():
            return gp_model(y).mean.sum(dim=0)

    def covar_fun(y):
        with gpytorch.settings.fast_pred_var():
            return gp_model(y).variance

    def get_mean_dy(y, create_graph=False):
        with gpytorch.settings.fast_pred_var():
            mean_dy = torch.autograd.functional.jacobian(
                mean_fun_sum, y, create_graph=create_graph
            )
        return mean_dy

    def gp_sensitivities(y):
        # evaluate GP part (GP jacobians)
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(to_tensor(y), requires_grad=True)
            # DERIVATIVE
            mean_dy = to_numpy(
                torch.autograd.functional.jacobian(mean_fun_sum, y_tensor)
            )

            with torch.no_grad():
                predictions = gp_model(
                    y_tensor
                )  # only model (we want to find true function)
                mean = to_numpy(predictions.mean)
                variance = to_numpy(predictions.variance)

        return mean, mean_dy, variance

    def P_propagation_with_y(y, P_vec, A_nom, create_graph=False):
        variance = covar_fun(y)
        mean_dy = get_mean_dy(y, create_graph=create_graph)
        # P_vec_tensor = to_tensor(P_vec)
        # P_next = P_propagation(P, B @ A_GP, B, torch.diag(variance[0,:]))

        # no diag needed for variance
        nx_nom = B_tensor.shape[0]
        nx_vec = int((nx_nom + 1) * nx_nom / 2)

        N = y.shape[0]
        P_vec_prop = torch.zeros((N, nx_vec))
        for i in range(N):
            A_GP = mean_dy[:, i, 0:nx_nom]
            P = vec2sym_mat(P_vec[i, :], nx_nom)
            A_prop = A_nom[i, :, :] + B_tensor @ A_GP
            P_prop = (
                A_prop @ P @ A_prop.T
                + B_tensor @ torch.diag(variance[i, :]) @ B_tensor.T
            )
            P_vec_prop[i, :] = sym_mat2vec(P_prop)

        return P_vec_prop

    def gp_sensitivities_with_prop(y, P, A_nom, A_nom_dy):
        # evaluate GP part (GP jacobians)
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(to_tensor(y), requires_grad=True)
            P_tensor = torch.autograd.Variable(to_tensor(P), requires_grad=True)
            A_tensor = torch.autograd.Variable(to_tensor(A_nom), requires_grad=True)
            A_dy_tensor = torch.autograd.Variable(
                to_tensor(A_nom_dy), requires_grad=False
            )
            # DERIVATIVE
            mean_dy = to_numpy(
                torch.autograd.functional.jacobian(mean_fun_sum, y_tensor)
            )
            prop_dy_partial = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda y: P_propagation_with_y(
                        y, P_tensor, A_tensor, create_graph=True
                    ).sum(dim=0),
                    y_tensor,
                )
            )
            prop_dA = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda A: P_propagation_with_y(
                        y_tensor, P_tensor, A, create_graph=True
                    ).sum(dim=0),
                    A_tensor,
                )
            )
            prop_dP = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda P: P_propagation_with_y(
                        y_tensor, P, A_tensor, create_graph=True
                    ).sum(dim=0),
                    P_tensor,
                )
            )
            prop_dA_dy = np.transpose(
                np.diagonal(
                    np.tensordot(prop_dA, A_nom_dy, ([2, 3], [1, 2])), axis1=1, axis2=2
                ),
                [0, 2, 1],
            )
            prop_dy = prop_dy_partial + prop_dA_dy

            with torch.no_grad():
                predictions = gp_model(
                    y_tensor
                )  # only model (we want to find true function)
                mean = to_numpy(predictions.mean)
                prop = to_numpy(P_propagation_with_y(y_tensor, P_tensor, A_tensor))

        return mean, mean_dy, prop, prop_dy, prop_dP

    if covar_jac:
        return gp_sensitivities_with_prop
    else:
        return gp_sensitivities


def vec2sym_mat(vec, nx):
    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx, nx))
        i, j = np.triu_indices(nx, m=nx)
    elif isinstance(vec, torch.Tensor):
        mat = torch.zeros((nx, nx), device=vec.device)
        i, j = torch.triu_indices(nx, nx)
    else:
        mat = SX.zeros(nx, nx)
        i, j = np.triu_indices(nx, m=nx)
    mat[i, j] = vec
    mat.T[i, j] = vec
    return mat


def sym_mat2vec(mat):
    nx = mat.shape[0]
    if isinstance(mat, np.ndarray):
        i, j = np.triu_indices(nx, m=nx)
        return mat[i, j]
    elif isinstance(mat, torch.Tensor):
        i, j = torch.triu_indices(nx, nx)
        return mat[i, j]
    elif isinstance(mat, cas.DM):
        mat_np = np.array(mat)
        i, j = np.triu_indices(nx, m=nx)
        return cas.DM(mat_np[i, j])
    else:
        i, j = np.triu_indices(nx, m=nx)
        return mat[i, j]
