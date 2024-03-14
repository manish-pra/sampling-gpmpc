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


def export_dempc_ocp(params):
    ocp = AcadosOcp()
    name_prefix = (
        "env_" + str(params["env"]["name"]) + "_i_" + str(params["env"]["i"]) + "_"
    )
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]

    const_expr, p = dempc_const_expr(
        x_dim, n_order, params
    )  # think of a pendulum and -pi/2,pi, pi/2 region is unsafe

    model = export_linear_model(name_prefix + "dempc", p, params)  # start pendulum at 0
    # model.con_h_expr = const_expr
    model_x = model.x
    model_u = model.u
    ocp.model = model

    ocp = dempc_cost_expr(ocp, model_x, model_u, x_dim, p, params)

    ocp = dempc_const_val(ocp, params, x_dim, n_order)

    # ocp = concat_const_val(ocp,params)

    ocp = dempc_set_options(ocp, params)

    return ocp


def dempc_const_expr(x_dim, n_order, params):
    # lb_cx_lin = ca.SX.sym('lb_cx_lin')
    # lb_cx_grad = ca.SX.sym('lb_cx_grad', x_dim, 1)
    # ub_cx_lin = ca.SX.sym('lb_cx_lin')
    # ub_cx_grad = ca.SX.sym('lb_cx_grad', x_dim, 1)
    xg = ca.SX.sym("xg", 1)
    we = ca.SX.sym("we", 1, 1)
    cw = ca.SX.sym("cw", 1, 1)

    # q_th = params["common"]["constraint"]
    # var = (ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)
    #         [:x_dim] - (lb_cx_lin + lb_cx_grad.T @ (model_x-x_lin)[:x_dim]))

    p_lin = ca.vertcat(xg, cw)
    # model.con_h_expr = ca.vertcat(lb_cx_lin +
    #                             lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th, cw*var)
    # model.con_h_expr_e = ca.vertcat(lb_cx_lin +
    #                                 lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    # model.p = ca.vertcat(model.p, p_lin)
    return 1, p_lin


def concat_const_val(ocp, params):
    x_dim = params["optimizer"]["x_dim"]
    if (
        params["algo"]["type"] == "ret_expander"
        or params["algo"]["type"] == "MPC_expander"
    ):
        lbx = np.array(params["optimizer"]["x_min"])[:x_dim]
        ubx = np.array(params["optimizer"]["x_max"])[:x_dim]
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]]), lbx]
        )
        ocp.constraints.ubu = np.concatenate(
            [ocp.constraints.ubu, np.array([1.0]), ubx]
        )
        ocp.constraints.idxbu = np.arange(ocp.constraints.idxbu.shape[0] + 1 + x_dim)
    else:
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]])]
        )
        ocp.constraints.ubu = np.concatenate([ocp.constraints.ubu, np.array([1.0])])
        ocp.constraints.idxbu = np.concatenate(
            [ocp.constraints.idxbu, np.array([ocp.model.u.shape[0] - 1])]
        )

    # ocp.constraints.x0 = np.concatenate(
    #     [ocp.constraints.x0, np.array([0.0])])

    ocp.constraints.lbx_e = np.concatenate([ocp.constraints.lbx_e, np.array([1.0])])
    ocp.constraints.ubx_e = np.concatenate([ocp.constraints.ubx_e, np.array([1.0])])
    ocp.constraints.idxbx_e = np.concatenate(
        [ocp.constraints.idxbx_e, np.array([ocp.model.x.shape[0] - 1])]
    )

    ocp.constraints.lbx = np.concatenate([ocp.constraints.lbx, np.array([0])])
    ocp.constraints.ubx = np.concatenate([ocp.constraints.ubx, np.array([2])])
    ocp.constraints.idxbx = np.concatenate(
        [ocp.constraints.idxbx, np.array([ocp.model.x.shape[0] - 1])]
    )
    return ocp


def dempc_cost_expr(ocp, model_x, model_u, x_dim, p, params):
    pos_dim = 1
    nx = params["agent"]["dim"]["nx"]
    q = 1e-3 * np.diag(np.ones(pos_dim))
    qx = np.diag(np.ones(pos_dim))
    xg = p[0]
    w = p[1]
    # cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    # either w_max can be decided here or outside
    # ocp.model.cost_expr_ext_cost =  w * \
    #     (model_x[:pos_dim] - xg).T @ qx @ (model_x[:pos_dim] - xg) + model_u.T @ (q) @ model_u
    # ocp.model.cost_expr_ext_cost_e =  w * (model_x[:pos_dim] - xg).T @ qx @ (model_x[:pos_dim] - xg)
    if params["optimizer"]["cost"] == "mean":
        ocp.model.cost_expr_ext_cost = (
            w * (model_x[:pos_dim] - xg).T @ qx @ (model_x[:pos_dim] - xg)
            + model_u.T @ (q) @ model_u
        )
        ocp.model.cost_expr_ext_cost_e = (
            w * (model_x[:pos_dim] - xg).T @ qx @ (model_x[:pos_dim] - xg)
        )
    else:
        ocp.model.cost_expr_ext_cost = (
            w * (model_x[::nx] - xg).T @ qx @ (model_x[::nx] - xg)
            + model_u.T @ (q) @ model_u
        )
        ocp.model.cost_expr_ext_cost_e = (
            w * (model_x[::nx] - xg).T @ qx @ (model_x[::nx] - xg)
        )

    # if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
    #     ocp.constraints.idxsh = np.array([1,2])
    #     ocp.cost.zl = 1e2 * np.array([1,1])
    #     ocp.cost.zu = 1e1 * np.array([1,0.1])
    #     ocp.cost.Zl = 1e1 * np.array([[1,0],[0,1]])
    #     ocp.cost.Zu = 1e1 * np.array([[1,0],[0,1]])
    # else:
    #     ocp.constraints.idxsh = np.array([1])
    #     ocp.cost.zl = 1e2 * np.array([1])
    #     ocp.cost.zu = 1e1 * np.array([1])
    #     ocp.cost.Zl = 1e1 * np.array([1])
    #     ocp.cost.Zu = 1e1 * np.array([1])

    # ocp.cost.cost_type = 'NONLINEAR_LS'
    # ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # ocp.cost.W_e = np.diag(1*np.ones(x_dim))
    # ocp.cost.W = np.diag(
    #     np.hstack([1*np.ide(x_dim), 1e-3*np.ones(x_dim), 1e-4]))
    # ocp.model.cost_y_expr = ca.vertcat(
    #     w*(model_x[:x_dim] - xg), model_u, w*var)
    # ocp.model.cost_y_expr_e = w*(model_x[:x_dim] - xg)
    # yref = np.zeros(2*x_dim+1)
    # ocp.cost.yref = yref
    # ocp.cost.yref_e = np.zeros(1*x_dim)
    return ocp


def dempc_const_val(ocp, params, x_dim, n_order):
    # constraints
    ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
    ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
    ocp.constraints.idxbu = np.arange(1)

    lbx = np.array(params["optimizer"]["x_min"] * params["agent"]["num_dyn_samples"])
    ubx = np.array(params["optimizer"]["x_max"] * params["agent"]["num_dyn_samples"])

    # lbx = params["optimizer"]["u_min"][0]*np.ones(n_order*x_dim)
    # lbx[:x_dim] = params["optimizer"]["x_min"]*np.ones(x_dim)

    # ubx = params["optimizer"]["u_max"][0]*np.ones(n_order*x_dim)
    # ubx[:x_dim] = params["optimizer"]["x_max"]*np.ones(x_dim)

    x0 = np.zeros(ocp.model.x.shape[0])
    x0 = np.array(
        params["env"]["start"] * params["agent"]["num_dyn_samples"]
    )  # np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()

    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    # ocp.constraints.lh = np.array([0, eps])
    # ocp.constraints.uh = np.array([10.0, 1e8])
    # ocp.constraints.lh_e = np.array([0.0])
    # ocp.constraints.uh_e = np.array([10.0])

    # ocp.constraints.lh = np.array([0, eps])
    # ocp.constraints.uh = np.array([10.0, 1.0e9])
    # lh_e = np.zeros(n_order*x_dim+1)
    # lh_e[0] = 0
    # ocp.constraints.lh_e = lh_e
    # uh_e = np.zeros(n_order*x_dim+1)
    # uh_e[0] = 10
    # ocp.constraints.uh_e = uh_e

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def dempc_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["Tf"]

    ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = 1.0e-1
    ocp.solver_options.integrator_type = "DISCRETE"  #'IRK'  # IRK , DISCRETE
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    # ocp.solver_options.tol = 1e-6
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.globalization = 'FIXED_STEP'
    # ocp.solver_options.alpha_min = 1e-2
    # ocp.solver_options.__initialize_t_slacks = 0
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.levenberg_marquardt = 1e-1
    # ocp.solver_options.print_level = 2
    # ocp.solver_options.qp_solver_iter_max = 400
    # ocp.solver_options.regularize_method = 'MIRROR'
    # ocp.solver_options.exact_hess_constr = 0
    # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
    # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
    # ocp.solver_options.eps_sufficient_descent = 5e-1
    # params = {'globalization': ['MERIT_BACKTRACKING', 'FIXED_STEP'],
    #       'line_search_use_sufficient_descent': [0, 1],
    #       'globalization_use_SOC': [0, 1]}
    return ocp


def export_sim(params, name):
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    sim = AcadosSim()
    # model = export_integrator_model(name)
    # model = export_n_integrator_model(name, n_order, x_dim)
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
    # sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.integrator_type = "ERK"
    sim.parameter_values = np.zeros(model.p.shape)
    # sim.solver_options.sim_method_num_stages = 2
    # sim.solver_options.sim_method_num_steps = 1  # 1
    # sim.solver_options.nlp_solver_tol_eq = 1e-9

    # set prediction horizon
    sim.solver_options.Tsim = params["optimizer"]["Tf"] / params["optimizer"]["H"]
    return sim


# def export_ocp(params):
#     ocp = AcadosOcp()
#     model = export_integrator_model()
#     h_lin = ca.SX.sym('h_lin')
#     h_grad = ca.SX.sym('h_grad')
#     x_lin = ca.SX.sym('x_lin')
#     p_lin = ca.vertcat(h_lin, h_grad, x_lin)
#     model.con_h_expr = h_lin + h_grad @ (model.x-x_lin)
#     model.con_h_expr_e = h_lin + h_grad @ (model.x-x_lin)
#     model.p = p_lin
#     ocp.model = model

#     # discretization
#     ocp.dims.N = params["optimizer"]["H"]
#     ocp.solver_options.tf = params["optimizer"]["Tf"]
#     q = np.array([-1.0])
#     # cost
#     ocp.cost.cost_type = 'EXTERNAL'
#     ocp.cost.cost_type_e = 'EXTERNAL'
#     ocp.model.cost_expr_ext_cost = model.x.T @ q @ model.x + \
#         model.u.T @ (-q) @ model.u
#     ocp.model.cost_expr_ext_cost_e = model.x.T @ q @ model.x

#     # constraints
#     ocp.constraints.lbu = np.array([-1.0])
#     ocp.constraints.ubu = np.array([1.0])
#     ocp.constraints.idxbu = np.array([0])
#     ocp.constraints.x0 = np.array([0.40])
#     ocp.constraints.lbx_e = np.array([0.40])
#     ocp.constraints.ubx_e = np.array([0.40])
#     ocp.constraints.idxbx_e = np.array([0])
#     # ocp.model.con_h_expr = np.array([1.05]) - (model.x-np.array([4.0]))**2
#     # ocp.model.con_h_expr_e = np.array([1.05]) - (model.x-np.array([4.0]))**2
#     ocp.constraints.lh = np.array([-1])
#     ocp.constraints.uh = np.array([10.0])
#     ocp.constraints.lh_e = np.array([-1])
#     ocp.constraints.uh_e = np.array([10.0])
#     ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
#     # set options
#     ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
#     # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
#     # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
#     ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
#     ocp.solver_options.integrator_type = 'IRK'  # IRK
#     # ocp.solver_options.print_level = 1
#     ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
#     return ocp
