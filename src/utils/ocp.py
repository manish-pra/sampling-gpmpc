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
    x_dim = params["agent"]["dim"]["nx"]

    const_expr, p = dempc_const_expr(
        x_dim, n_order, params
    )  # think of a pendulum and -pi/2,pi, pi/2 region is unsafe

    model = export_linear_model(name_prefix + "dempc", p, params)  # start pendulum at 0
    nx = params["agent"]["dim"]["nx"]
    model_x = model.x
    model_u = model.u

    if params["env"]["dynamics"] == "bicycle":
        num_dyn = params["agent"]["num_dyn_samples"]
        const_expr = []
        for ellipse in params["env"]["ellipses"]:
            x0 = params["env"]["ellipses"][ellipse][0]
            y0 = params["env"]["ellipses"][ellipse][1]
            a = params["env"]["ellipses"][ellipse][2]
            b = params["env"]["ellipses"][ellipse][3]
            f = params["env"]["ellipses"][ellipse][4]
            for i in range(num_dyn):
                expr = (model_x[nx * i] - x0).T @ (model_x[nx * i] - x0) / a + (
                    model_x[nx * i + 1] - y0
                ).T @ (model_x[nx * i + 1] - y0) / b
                const_expr = ca.vertcat(const_expr, expr)
        model.con_h_expr = const_expr
        model.con_h_expr_e = const_expr
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


def dempc_cost_expr(ocp, model_x, model_u, x_dim, p, params):
    pos_dim = 1
    nx = params["agent"]["dim"]["nx"]
    nu = params["agent"]["dim"]["nu"]
    Qu = np.diag(np.array(params["optimizer"]["Qu"]))
    xg = np.array(params["env"]["goal_state"])
    w = params["optimizer"]["w"]
    Qx = np.diag(np.array(params["optimizer"]["Qx"]))

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
            (model_x[nx * i : nx * (i + 1)] - xg).T
            @ Qx
            @ (model_x[nx * i : nx * (i + 1)] - xg)
        )
    ocp.model.cost_expr_ext_cost = expr / ns + model_u.T @ (Qu) @ model_u
    ocp.model.cost_expr_ext_cost_e = expr / ns

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
    ocp.constraints.idxbu = np.arange(ocp.constraints.ubu.shape[0])

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

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    if params["env"]["dynamics"] == "bicycle":
        nh = params["agent"]["num_dyn_samples"] * len(params["env"]["ellipses"])
        f = params["env"]["ellipses"]["n1"][4]
        ocp.constraints.lh = np.array([f] * nh)
        ocp.constraints.uh = np.array([1e3] * nh)
        ocp.constraints.lh_e = np.array([f] * nh)
        ocp.constraints.uh_e = np.array([1e3] * nh)

        nbx = 0
        # nbx = len(lbx)
        # ocp.constraints.idxsbx = np.arange(nbx)
        # ocp.constraints.idxsbx_e = np.arange(nbx)
        ocp.constraints.idxsh = np.arange(nh)
        ocp.constraints.idxsh_e = np.arange(nh)

        ns = nh + nbx
        ocp.cost.zl = 1e3 * np.array([1] * ns)
        ocp.cost.zu = 1e3 * np.array([1] * ns)
        ocp.cost.Zl = 1e3 * np.array([1] * ns)
        ocp.cost.Zu = 1e3 * np.array([1] * ns)

        ocp.cost.zl_e = 1e3 * np.array([1] * ns)
        ocp.cost.zu_e = 1e3 * np.array([1] * ns)
        ocp.cost.Zl_e = 1e3 * np.array([1] * ns)
        ocp.cost.Zu_e = 1e3 * np.array([1] * ns)

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
    ocp.solver_options.tf = params["optimizer"]["dt"] * params["optimizer"]["H"]
    # ocp.solver_options.Tsim = params["optimizer"]["dt"]

    # ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.hessian_approx = "EXACT"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = params["optimizer"]["options"][
        "levenberg_marquardt"
    ]
    ocp.solver_options.integrator_type = "DISCRETE"  #'IRK'  # IRK , DISCRETE
    # ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    # ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP
    # ocp.solver_options.rti_log_residuals = 1
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.tol = 1e-6
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.globalization = (
    #     "MERIT_BACKTRACKING"  # 'MERIT_BACKTRACKING', 'FIXED_STEP'
    # )
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
