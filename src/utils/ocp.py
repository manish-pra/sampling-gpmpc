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
    u_dim = params["agent"]["dim"]["nu"]

    xg = ca.SX.sym("xg", 1)
    # we = ca.SX.sym("we", 1, 1)
    cw = ca.SX.sym("cw", 1, 1)
    tightening = ca.SX.sym("tilde_eps_i", x_dim + u_dim + 1, 1) #[tightening, tilde_eps]

    p= ca.vertcat(xg, cw, tightening)

    # const_expr, p = dempc_const_expr(
    #     x_dim, n_order, params
    # )  # think of a pendulum and -pi/2,pi, pi/2 region is unsafe

    model = export_linear_model(name_prefix + "dempc", p, params)  # start pendulum at 0
    nx = params["agent"]["dim"]["nx"]
    model_x = model.x
    model_u = model.u
    num_dyn = params["agent"]["num_dyn_samples"]
    if "bicycle" in params["env"]["dynamics"]:
        const_expr = []
        const_expr_e = []
        if params["env"]["ellipses"]:
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
            const_expr_e = const_expr
        if params["agent"]["tight"]["use"]:
            tighten_ns = ca.repmat(tightening[:x_dim], num_dyn)
            con_h_expr = ca.vertcat(con_h_expr, model_x - tighten_ns, model_x + tighten_ns)
            con_h_expr_e = ca.vertcat(con_h_expr_e , model_x - tighten_ns, model_x + tighten_ns)
        if params["agent"]["feedback"]["use"]:
            x_equi = np.array(params["env"]["goal_state"])
            K = np.array(params["optimizer"]["terminal_tightening"]["K"])
            u_expr = []
            for i in range(num_dyn):
                expr = K@(x_equi - model_x[nx * i : nx * (i + 1)]) + model_u #+ tightening[x_dim]
                u_expr = ca.vertcat(u_expr, expr)
            for i in range(num_dyn):
                expr = K@(x_equi - model_x[nx * i : nx * (i + 1)]) + model_u #- tightening[x_dim]
                u_expr = ca.vertcat(u_expr, expr)
            const_expr = ca.vertcat(const_expr, u_expr)
        model.con_h_expr = const_expr
        model.con_h_expr_e = const_expr_e
    if params["env"]["dynamics"] == "Pendulum1D":
        x_equi = np.array(params["env"]["goal_state"])
        # tilde_eps_i = p[2]
        u_expr = []
        K = np.array(params["optimizer"]["terminal_tightening"]["K"])
        for i in range(num_dyn):
            expr = K@(x_equi - model_x[nx * i : nx * (i + 1)]) + model_u + tightening[x_dim]
            u_expr = ca.vertcat(u_expr, expr)
        for i in range(num_dyn):
            expr = K@(x_equi - model_x[nx * i : nx * (i + 1)]) + model_u - tightening[x_dim]
            u_expr = ca.vertcat(u_expr, expr)
        tighten_ns = ca.repmat(tightening[:x_dim], num_dyn)
        model.con_h_expr = ca.vertcat(model_x - tighten_ns, model_x + tighten_ns, u_expr)

        const_expr = []
        for i in range(num_dyn):
            xf = np.array(params["env"]["goal_state"])
            expr = (
                (model_x[nx * i : nx * (i + 1)] - xf).T
                @ np.array(params["optimizer"]["terminal_tightening"]["P"])
                @ (model_x[nx * i : nx * (i + 1)] - xf)
            )
            const_expr = ca.vertcat(const_expr, expr)

        model.con_h_expr_e = const_expr

    ocp.model = model
    ocp = dempc_cost_expr(ocp, model_x, model_u, x_dim, p, params)

    ocp = dempc_const_val(ocp, params, x_dim, n_order, p)
    ocp = dempc_set_options(ocp, params)

    return ocp


# def dempc_const_expr(x_dim, n_order, params):
#     xg = ca.SX.sym("xg", 1)
#     we = ca.SX.sym("we", 1, 1)
#     cw = ca.SX.sym("cw", 1, 1)
#     tilde_eps_i = ca.SX.sym("tilde_eps_i", 1, 1)

#     p_lin = ca.vertcat(xg, cw, tilde_eps_i)
#     return 1, p_lin


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

    return ocp


def dempc_const_val(ocp, params, x_dim, n_order, p):
    tilde_eps_i = p[2]
    ns = params["agent"]["num_dyn_samples"]
    # constraints
    if params["agent"]["feedback"]["use"]:
        ocp.constraints.lbu = np.array(params["agent"]["feedback"]["v_min"])
        ocp.constraints.ubu = np.array(params["agent"]["feedback"]["v_max"])
    else:
        ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
    ocp.constraints.idxbu = np.arange(ocp.constraints.ubu.shape[0])

    lbx = np.array(params["optimizer"]["x_min"] * ns) 
    ubx = np.array(params["optimizer"]["x_max"] * ns) 

    x0 = np.zeros(ocp.model.x.shape[0])
    x0 = np.array(params["env"]["start"] * ns)  # np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    if params["env"]["dynamics"] == "Pendulum1D":
        ocp.constraints.lh = np.hstack([lbx, lbx, params["optimizer"]["u_min"]*2*ns])
        ocp.constraints.uh = np.hstack([ubx, ubx, params["optimizer"]["u_max"]*2*ns])
        # ocp.constraints.lh = np.hstack([lbx, lbx])
        # ocp.constraints.uh = np.hstack([ubx, ubx])
        delta = params["optimizer"]["terminal_tightening"]["delta"]
        ocp.constraints.lh_e = np.hstack([[0] * ns])
        ocp.constraints.uh_e = np.hstack([[delta] * ns])
        # size = len(ocp.constraints.lh)
        # ocp.cost.Zl = 1e6 * np.array([1] * size)
        # ocp.cost.Zu = 1e5 * np.array([1] * size)
        # ocp.cost.zl = 1e6 * np.array([1] * size)
        # ocp.cost.zu = 1e5 * np.array([1] * size)
        # ocp.constraints.idxsh = np.arange(size_e)
        size_e = len(ocp.constraints.lh_e)
        ocp.cost.Zl_e = 1e6 * np.array([1] * size_e)
        ocp.cost.Zu_e = 1e5 * np.array([1] * size_e)
        ocp.cost.zl_e = 1e6 * np.array([1] * size_e)
        ocp.cost.zu_e = 1e5 * np.array([1] * size_e)
        ocp.constraints.idxsh_e = np.arange(size_e)
    # ocp.constraints.idxsh = np.arange(2 * lbx.shape[0])

    if "bicycle" in params["env"]["dynamics"]:
        if params["env"]["ellipses"]:
            nh = params["agent"]["num_dyn_samples"] * len(params["env"]["ellipses"])
            f = params["env"]["ellipses"]["n1"][4]
            if params["agent"]["tight"]["use"]:
                ocp.constraints.lh = np.hstack(
                    [[f] * nh, lbx, lbx, params["optimizer"]["u_min"]*2*ns]
                )  # np.array([f] * nh)
                ocp.constraints.uh = np.hstack(
                    [[1e8] * nh, ubx, ubx, params["optimizer"]["u_max"]*2*ns]
                )  # np.array([1e3] * nh)
                ocp.constraints.lh_e = np.hstack(
                    [[f] * nh, lbx, lbx]
                )  # np.array([f] * nh)
                ocp.constraints.uh_e = np.hstack(
                    [[1e8] * nh, ubx, ubx]
                )  # np.array([1e3] * nh)
            elif params["agent"]["feedback"]["use"]:
                ocp.constraints.lh = np.hstack(
                    [[f] * nh, params["optimizer"]["u_min"]*2*ns]
                )
                ocp.constraints.uh = np.hstack(
                    [[1e3] * nh, params["optimizer"]["u_max"]*2*ns]
                )
                ocp.constraints.lh_e = np.hstack([[f] * nh])
                ocp.constraints.uh_e = np.hstack([[1e3] * nh]
                )
            else:
                ocp.constraints.lh = np.array([f] * nh)
                ocp.constraints.uh = np.array([1e3] * nh)
                ocp.constraints.lh_e = np.array([f] * nh)
                ocp.constraints.uh_e = np.array([1e3] * nh)

            # nbx = 0
            nbx = len(lbx)
            ocp.constraints.idxsbx = np.arange(nbx)
            ocp.constraints.idxsbx_e = np.arange(nbx)
            ocp.constraints.idxsh = np.arange(nh)
            ocp.constraints.idxsh_e = np.arange(nh)

            ns = nh + nbx
            ocp.cost.zl = 1e6 * np.array([1] * ns)
            ocp.cost.zu = 1e5 * np.array([1] * ns)
            ocp.cost.Zl = 1e6 * np.array([1] * ns)
            ocp.cost.Zu = 1e5 * np.array([1] * ns)

            ocp.cost.zl_e = 1e6 * np.array([1] * ns)
            ocp.cost.zu_e = 1e5 * np.array([1] * ns)
            ocp.cost.Zl_e = 1e6 * np.array([1] * ns)
            ocp.cost.Zu_e = 1e5 * np.array([1] * ns)

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def dempc_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["dt"] * params["optimizer"]["H"]

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = params["optimizer"]["options"][
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
