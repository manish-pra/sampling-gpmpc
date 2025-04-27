import numpy as np

def get_reachable_set_ball(params, V_k, eps_vec=None):
    H = params["optimizer"]["H"]
    assert V_k.shape[0] == H + 1
    # computation of tightenings
    P = np.array(params["optimizer"]["terminal_tightening"]["P"])
    # P *=10
    print(f"P = {P}", "V_k = ", V_k)
    L = params["agent"]["tight"]["Lipschitz"]
    dyn_eps = params["agent"]["tight"]["dyn_eps"]
    w_bound = params["agent"]["tight"]["w_bound"]
    var_eps = (dyn_eps + w_bound)
    if eps_vec is not None:
        # np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([8e-4,9e-4,3e-4]))
        # B_d_norm = (np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([3.65e-4,4e-4,1.35e-4]))/var_eps)*V_k
        B_d_norm = (np.dot(np.sqrt(np.diag(P[:3][:3])),eps_vec)/var_eps)*V_k
    else:
        B_d_norm = np.sum(np.sqrt(np.diag(P[:3][:3])))*V_k
    P_inv = np.linalg.inv(P)
    K = np.array(params["optimizer"]["terminal_tightening"]["K"])
    B_eps_0 = 0
    tightenings = np.sqrt(np.diag(P_inv))*B_eps_0
    u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_0
    tilde_eps_list = []
    tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_0]]))
    ci_list = []
    for stage in range(1, H + 1):
        B_eps_k = var_eps*B_d_norm[stage-1] * np.sum(np.power(L, np.arange(0, stage)))  
        # arange has inbuild -1 in [sstart, end-1]
        # box constraints tightenings
        tightenings = np.sqrt(np.diag(P_inv))*B_eps_k
        u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_k
        print(f"u_tight_{stage} = {u_tight}")
        tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_k]]))
        ci_list.append(B_eps_k)
        print(f"tilde_eps_{stage} = {tilde_eps_list[-1]}")
    # quit()
    return tilde_eps_list, ci_list