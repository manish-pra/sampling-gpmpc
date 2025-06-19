import casadi as ca
import numpy as np
import CarDynamics

class MPCC:
    def __init__(self, model, num_samples,weight_samples = None, horizon=10, lambda_slack=10):
        self.mdl, self.H, self.dt = model, horizon, model.dt
        self.weight_samples = weight_samples
        self.Nw = num_samples
        self.q_c, self.q_l = 3.0, 100
        self.gamma = 3.0
        self.Ru = np.diag([ 1.0, 1.0]) 
        self.Rdu = np.diag([50.0, 50.0])
        self.Rv = 0.05                
        self.Rdv = 1000
        self.lambda_slack = lambda_slack
        self.bounds_state = dict(x=(0, 100), y=(-2, 6),
                                 psi=(-1.2, 1.2), vx=(1.5, 8.0),
                                 vy=(-2, 2), omega=(-2, 2))
        self.bounds_ctrl = dict(delta=(-0.45, 0.45), tau=(-0.02, 0.02))
        self.v_bounds = (0.0, 8.0)
        self.obstacles = [(20, 1.95), (40, 1.95)]

        self._build_optimizer()
    def _build_optimizer(self):
        Nw, H = self.Nw, self.H
        opti = ca.Opti(); self.opti = opti

        X = {s: opti.variable(6, H+1) for s in range(Nw)}
        U = opti.variable(2, H)
        Theta = opti.variable(1, H+1)
        V = opti.variable(1, H)
        s_obs = opti.variable(len(self.obstacles), H+1); self.s_obs = s_obs

        x0_p, th0_p = opti.parameter(6), opti.parameter()
        u_prev_p = opti.parameter(2) 
        v_prev_p = opti.parameter() 
        self.x0, self.th0 = x0_p, th0_p
        self.u_prev_p, self.v_prev_p = u_prev_p, v_prev_p
        self.X, self.U, self.Theta, self.V = X, U, Theta, V
        cost = 0
        for s in range(Nw):
            for k in range(H):
                e_c = CarDynamics.StraightTrack.contouring_error(X[s][1, k])
                e_l = CarDynamics.StraightTrack.lag_error(Theta[0, k], X[s][0, k])
                cost += self.q_c*e_c**2 + self.q_l*e_l**2
                cost -= self.gamma*V[0, k]*self.dt          
        cost /= Nw
        for k in range(H):
            if k == 0:
                du = U[:,k] - u_prev_p
                dv = V[0,k] - v_prev_p
            else:
                du = U[:,k] - U[:,k-1]
                dv = V[0,k] - V[0,k-1]
            cost += ca.mtimes([du.T, self.Rdu, du]) + self.Rdv*dv**2
            cost += ca.mtimes([U[:,k].T, self.Ru, U[:,k]]) + self.Rv*V[0,k]**2

        cost += self.lambda_slack*ca.sumsqr(s_obs)
        opti.minimize(cost)

        opti.subject_to(Theta[0,0] == th0_p)
        for s in range(Nw):
            opti.subject_to(X[s][:,0] == x0_p)


        if self.weight_samples is None:
            for k in range(H):
                for s in range(Nw):
                    opti.subject_to(X[s][:,k+1] == self.mdl.step(X[s][:,k], U[:,k], self.mdl.Df, self.mdl.Dr, ca))
                opti.subject_to(Theta[0,k+1] == Theta[0,k] + V[0,k]*self.dt)
        else:
            for k in range(H):
                for s in range(Nw):
                    Df_s, Dr_s = self.weight_samples[s][0], self.weight_samples[s][1]
                    opti.subject_to(X[s][:,k+1] == self.mdl.step(X[s][:,k], U[:,k], Df_s, Dr_s, ca))
                opti.subject_to(Theta[0,k+1] == Theta[0,k] + V[0,k]*self.dt)

        bx, by = self.bounds_state['x'], self.bounds_state['y']
        bpsi, bvx = self.bounds_state['psi'], self.bounds_state['vx']
        bvy, bomega = self.bounds_state['vy'], self.bounds_state['omega']
        bdelta, btau = self.bounds_ctrl['delta'], self.bounds_ctrl['tau']


        for k in range(H+1):
            for s in range(Nw):
                opti.subject_to([bx[0] <= X[s][0,k], X[s][0,k] <= bx[1]])
                opti.subject_to([by[0] <= X[s][1,k], X[s][1,k] <= by[1]])
                opti.subject_to([bpsi[0] <= X[s][2,k],X[s][2,k] <= bpsi[1]])
                opti.subject_to([bvx[0] <= X[s][3,k], X[s][3,k] <= bvx[1]])
                opti.subject_to([bvy[0] <= X[s][4,k], X[s][4,k] <= bvy[1]])
                opti.subject_to([bomega[0] <= X[s][5,k],X[s][5,k] <= bomega[1]])

                for j,(ox,oy) in enumerate(self.obstacles):
                    dist = ((X[s][0,k]-ox)**2)/9 + (X[s][1,k]-oy)**2
                    opti.subject_to(dist + s_obs[j,k] >= 5.67)

        for k in range(H):
            opti.subject_to([bdelta[0] <= U[0,k], U[0,k] <= bdelta[1]])
            opti.subject_to([ btau[0] <= U[1,k], U[1,k] <= btau[1]])
            opti.subject_to([self.v_bounds[0] <= V[0,k], V[0,k] <= self.v_bounds[1]])

        opti.subject_to(ca.vec(s_obs) >= 0)

        self.opti.solver('ipopt', {'ipopt.print_level':0,
                              'print_time':0,
                              'ipopt.check_derivatives_for_naninf':'yes',
                              'ipopt.tol':1e-6})

    def solve(self, x_curr, theta_curr, warm=None):
       
        self.opti.set_value(self.x0, x_curr)
        self.opti.set_value(self.th0, theta_curr)

        if warm is None:
            self.opti.set_value(self.u_prev_p, np.zeros(2))
            self.opti.set_value(self.v_prev_p, x_curr[3])

            for s in range(self.Nw):
                self.opti.set_initial(
                    self.X[s],
                    np.repeat(x_curr[:, None], self.H + 1, axis=1)
                )
        else:
            Xw_dict, Uw, Vw, Thw, Sw = warm        
            current_x = x_curr
            self.opti.set_value(self.u_prev_p, Uw[:, 0])
            self.opti.set_value(self.v_prev_p, Vw.flatten()[0])

            for s in range(self.Nw):
                X_shift = np.hstack([Xw_dict[s][:, 1:], Xw_dict[s][:, -1:]])
                X_shift[:, 0] = x_curr
                self.opti.set_initial(self.X[s], X_shift)


            U_shift = np.hstack([Uw[:, 1:], Uw[:, -1:]])
            self.opti.set_initial(self.U, U_shift)

            V_shift = np.hstack([Vw[:, 1:], Vw[:, -1:]])
            self.opti.set_initial(self.V, V_shift)

            Th_shift = np.hstack([Thw[:, 1:], Thw[:, -1:]])
            self.opti.set_initial(self.Theta, Th_shift)

            S_shift = np.hstack([Sw[:, 1:], Sw[:, -1:]])
            self.opti.set_initial(self.s_obs, S_shift) 
        try:
            sol = self.opti.solve()
            X_opt = {s: sol.value(self.X[s]) for s in range(self.Nw)}
            Uopt = np.asarray(sol.value(self.U))                      
            Vopt = np.asarray(sol.value(self.V)).reshape(1,self.H)
            Thopt = np.asarray(sol.value(self.Theta)).reshape(1, self.H+1)
            Sopt  = np.asarray(sol.value(self.s_obs)).reshape(len(self.obstacles), self.H+1)
            return X_opt, Uopt, Vopt, Thopt, Sopt
        except RuntimeError as e:
            print("Ipopt raised:",e)
            X_samples_last = []
            for s in range(self.Nw):
                X_last = np.asarray(self.opti.debug.value(self.X[s]))
                X_samples_last.append(X_last)
                print(X_last)
            U_last = np.asarray(self.opti.debug.value(self.U))
            print(U_last)
            self.opti.debug.show_infeasibilities(1e-6)
            return None, X_samples_last, U_last
