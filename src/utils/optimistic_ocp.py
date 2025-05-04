import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_sim import AcadosSim
from src.utils.model import (
    export_integrator_model,
    export_n_integrator_model,
    export_linear_model,
    export_pendulum_ode_model_with_discrete_rk4,
    export_pendulum_ode_model_with_discrete_rk4_Lc,
    export_NH_integrator_ode_model_with_discrete_rk4,
    export_robot_model_with_discrete_rk4,
    export_bicycle_model_with_discrete_rk4,
    export_unicycle_model_with_discrete_rk4,
)
import copy

def export_optimistic_ocp(params, env_ocp_handler=None):
    ocp = AcadosOcp()
    name_prefix = (
        "env_" + str(params["env"]["name"]) + "_i_" + str(params["env"]["i"]) + "_"
    )
    n_order = params["optimizer"]["order"]
    x_dim = params["agent"]["dim"]["nx"]

    _, p = optimistic_const_expr(
        x_dim, n_order, params
    )  # think of a pendulum and -pi/2,pi, pi/2 region is unsafe
    
    optimistic_params = copy.deepcopy(params)
    optimistic_params["agent"]["num_dyn_samples"] = 1
    optimistic_params["agent"]["dim"]["nu"] += params["agent"]["dim"]["nx"]

    model = export_linear_model(name_prefix + "optimistic", p, optimistic_params)  # start pendulum at 0
    nx = optimistic_params["agent"]["dim"]["nx"]
    model_x = model.x
    model_u = model.u

    const_expr = env_ocp_handler("const_expr", model_x, 1)
    model.con_h_expr_e = const_expr
    ocp.model = model
    ocp = optimistic_cost_expr(ocp, model_x, model_u, x_dim, p, optimistic_params)

    ocp = optimistic_const_val(ocp, optimistic_params, x_dim, n_order, p, env_ocp_handler)
    ocp = optimistic_set_options(ocp, optimistic_params)

    return ocp


def optimistic_const_expr(x_dim, n_order, params):
    xg = ca.SX.sym("xg", 1)
    we = ca.SX.sym("we", 1, 1)
    cw = ca.SX.sym("cw", 1, 1)
    tilde_eps_i = ca.SX.sym("tilde_eps_i", 1, 1)

    p_lin = ca.vertcat(xg, cw, tilde_eps_i)
    return 1, p_lin


def optimistic_cost_expr(ocp, model_x, model_u, x_dim, p, params):
    pos_dim = 1
    nx = params["agent"]["dim"]["nx"]
    nu = params["agent"]["dim"]["nu"]
    Qu = np.diag(np.array(params["optimistic_optimizer"]["Qu"]))
    xg = np.array(params["env"]["goal_state"])
    xg_dim = xg.shape[0]
    w = params["optimistic_optimizer"]["w"]
    Qx = np.diag(np.array(params["optimistic_optimizer"]["Qx"]))

    # cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    if params["optimizer"]["cost"] == "mean":
        ns = 1
    else:
        ns = params["agent"]["num_dyn_samples"]
    expr = 0

    for i in range(ns):
        expr += (
            (model_x[nx * i : nx * (i + 1)][:xg_dim] - xg).T
            @ Qx
            @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - xg)
        )
    ocp.model.cost_expr_ext_cost = expr / ns + model_u.T @ (Qu) @ model_u
    ocp.model.cost_expr_ext_cost_e = expr / ns

    return ocp


def optimistic_const_val(ocp, params, x_dim, n_order, p, env_ocp_handler):
    tilde_eps_i = p[2]
    ns = params["agent"]["num_dyn_samples"]
    # constraints
    ocp.constraints.lbu = np.array(params["optimistic_optimizer"]["u_min"])
    ocp.constraints.ubu = np.array(params["optimistic_optimizer"]["u_max"])
    ocp.constraints.idxbu = np.arange(ocp.constraints.ubu.shape[0])

    lbx = np.array(params["optimistic_optimizer"]["x_min"] * ns)
    ubx = np.array(params["optimistic_optimizer"]["x_max"] * ns)

    x0 = np.zeros(ocp.model.x.shape[0])
    x0 = np.array(params["env"]["start"] * ns)  # np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    lh, uh, lh_e, uh_e = env_ocp_handler("const_value", 1)
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_e = lh_e
    ocp.constraints.uh_e = uh_e

    # size_e = len(ocp.constraints.lh_e)
    # ocp.cost.Zl_e = 1e7 * np.array([1] * size_e)
    # ocp.cost.Zu_e = 1e6 * np.array([1] * size_e)
    # ocp.cost.zl_e = 1e7 * np.array([1] * size_e)
    # ocp.cost.zu_e = 1e6 * np.array([1] * size_e)
    # ocp.constraints.idxsh_e = np.arange(size_e)

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def optimistic_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["dt"] * params["optimizer"]["H"]

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = params["optimistic_optimizer"]["options"][
        "levenberg_marquardt"
    ]
    ocp.solver_options.integrator_type = "DISCRETE"  #'IRK'  # IRK , DISCRETE
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.qp_solver_warm_start = 1
    return ocp


def export_sim(params, name):
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    sim = AcadosSim()
    model = export_pendulum_ode_model_with_discrete_rk4("oracle_sim", n_order, x_dim)
    h_lin = ca.SX.sym("h_lin")
    h_grad = ca.SX.sym("h_grad")
    x_lin = ca.SX.sym("x_lin")
    p_lin = ca.vertcat(h_lin, h_grad, x_lin)
    model.con_h_expr = h_lin + h_grad @ (model.x - x_lin)
    model.con_h_expr_e = h_lin + h_grad @ (model.x - x_lin)
    model.p = p_lin
    sim.model = model
    # solver options
    sim.solver_options.integrator_type = "ERK"
    sim.parameter_values = np.zeros(model.p.shape)

    # set prediction horizon
    sim.solver_options.Tsim = params["optimizer"]["Tf"] / params["optimizer"]["H"]
    return sim
