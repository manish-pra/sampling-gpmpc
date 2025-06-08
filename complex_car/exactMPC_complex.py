import numpy as np, time, casadi as ca, matplotlib.pyplot as plt
import matplotlib.patches as patches

class CarDynamicsComplex:
    def __init__(self, dt=0.06, lf=1.105, lr=1.738, Cf = 1.2, Bf = 6.0, Cr = 1.2, Br = 6.0, Iz = 1.6*1e-4, m = 0.03, Df = 0.84, Dr = 0.84):
        self.dt = dt
        self.lf = lf
        self.lr = lr
        self.Cf = Cf
        self.Cr = Cr
        self.Iz = 0.5*m*(lf**2+lr**2)
        self.Bf = Bf
        self.Br = Br
        self.m = m
        self.Df = Df
        self.Dr = Dr

    def step(self, x, u):
        xpos, ypos, psi, vx, vy, omega = x
        delta, tau = u
        alphaf = np.arctan((vy + self.lf * omega) / vx) - delta
        alphar = np.arctan((vy - self.lr * omega) / vx)
        Ff = self.Df * np.sin(self.Cf* np.arctan(self.Bf * alphaf))
        Fr = self.Dr * np.sin(self.Cr* np.arctan(self.Br * alphar))
        Fx = tau
        x_next = x + self.dt* np.array([vx * np.cos(psi) - vy * np.sin(psi),
                           vx * np.sin(psi) + vy * np.cos(psi),
                           omega,
                           (Fx - Ff * np.sin(delta) + self.m * vy * omega) / self.m,
                           (Fr + Ff *np.cos(delta) - self.m * vx * omega) / self.m,
                           (Ff * self.lf * np.cos(delta) - Fr * self.lr) / self.Iz])
        
        return x_next

    def casadi_step(self, x, u):
        xpos, ypos, psi, vx, vy, omega = x[0], x[1], x[2], x[3], x[4], x[5]
        delta, tau = u[0], u[1]
        alphaf = ca.atan((vy + self.lf * omega) / vx) - delta
        alphar = ca.atan((vy - self.lr * omega) / vx)
        Ff = self.Df * ca.sin(self.Cf* ca.atan(self.Bf * alphaf))
        Fr = self.Dr * ca.sin(self.Cr* ca.atan(self.Br * alphar))
        Fx = tau 
        x_dot = ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi),
                           vx * ca.sin(psi) + vy * ca.cos(psi),
                           omega,
                           (Fx - Ff * ca.sin(delta) + self.m * vy * omega) / self.m,
                           (Fr + Ff * ca.cos(delta) - self.m * vx * omega) / self.m,
                           (Ff * self.lf * ca.cos(delta) - Fr * self.lr) / self.Iz)
        return x + self.dt * x_dot

class ExactMPC:
    def __init__(self, model, horizon = 16, lambda_slack=100):
        self.mdl, self.H = model, horizon
        self.lambda_slack = lambda_slack

        self.Q = np.diag([ 2, 36, 0.01, 0.02, 0.02, 0.02])
        self.R = np.diag([ 2,  2])

        self.x_ref, self.y_ref = 70.0, 1.95
        self.bounds_state = dict(x=(0,100), y=(0.0,6.0),
                                 psi=(-1.2,1.2), vx=(0.5,8.0), vy=(-1.0, 1.0), omega=(-2.0,2.0))
        self.bounds_ctrl  = dict(delta =(-4.0,4.0), tau=(-3.0,3.0))
        self.obstacles = [(20,2), (40,2)]


        self._build_optimizer()

    def _build_optimizer(self):
        self.opti = ca.Opti()
        self.X = self.opti.variable(6, self.H+1)
        self.U = self.opti.variable(2, self.H)
        self.s_obs = self.opti.variable(len(self.obstacles), self.H+1)
        self.x0 = self.opti.parameter(6)
        
        cost = 0
        for k in range(self.H):
            e = self.X[:,k] - ca.vertcat(self.x_ref, self.y_ref, 0, 0, 0, 0)
            cost += ca.mtimes([e.T, self.Q, e])
            cost += ca.mtimes([self.U[:,k].T, self.R, self.U[:,k]])
        cost += self.lambda_slack * ca.sumsqr(self.s_obs)
        self.opti.minimize(cost)

        self.opti.subject_to(self.X[:,0] == self.x0)
        for k in range(self.H):
            x_next = self.mdl.casadi_step(self.X[:,k], self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == x_next)

        bx, by = self.bounds_state['x'], self.bounds_state['y']
        bpsi, bvx, bvy, bomega = self.bounds_state['psi'], self.bounds_state['vx'], self.bounds_state['vy'], self.bounds_state['omega']
        bdelta, btau = self.bounds_ctrl['delta'], self.bounds_ctrl['tau']

        for k in range(self.H+1):
            self.opti.subject_to(self.X[0,k] >= bx[0]); self.opti.subject_to(self.X[0,k] <= bx[1])
            self.opti.subject_to(self.X[1,k] >= by[0]); self.opti.subject_to(self.X[1,k] <= by[1])
            self.opti.subject_to(self.X[2,k] >= bpsi[0]); self.opti.subject_to(self.X[2,k] <= bpsi[1])
            self.opti.subject_to(self.X[3,k] >= bvx[0]); self.opti.subject_to(self.X[3,k] <= bvx[1])
            self.opti.subject_to(self.X[4,k] >= bvy[0]); self.opti.subject_to(self.X[4,k] <= bvy[1])
            self.opti.subject_to(self.X[5,k] >= bomega[0]); self.opti.subject_to(self.X[5,k] <= bomega[1])
            for j,(ox,oy) in enumerate(self.obstacles):
                dist = ((self.X[0,k]-ox)**2)/9 + (self.X[1,k]-oy)**2
                self.opti.subject_to(dist + self.s_obs[j,k] >= 5.67)

        for k in range(self.H):
            self.opti.subject_to(self.U[0,k] >= bdelta[0]); self.opti.subject_to(self.U[0,k] <= bdelta[1])
            self.opti.subject_to(self.U[1,k] >= btau[0]); self.opti.subject_to(self.U[1,k] <= btau[1])

        self.opti.subject_to(ca.vec(self.s_obs) >= 0)
        opts = {
    'ipopt.constr_viol_tol': 1e-6,
    'ipopt.acceptable_constr_viol_tol': 1e-6,
    'ipopt.tol': 1e-6,
    'ipopt.print_level': 0,
    'print_time': 0
}
        self.opti.solver('ipopt', opts)
    
    def solve(self, current_state, x_warm = None, u_warm = None):
        self.opti.set_value(self.x0, current_state)
        if u_warm is not None:
            u_shifted = np.hstack([u_warm[:, 1:], u_warm[:, -1:]])
            X_shift = np.hstack([x_warm[:,1:], x_warm[:,-1:]])  
            self.opti.set_initial(self.U, u_shifted)
            self.opti.set_initial(self.X, X_shift)
        else:
            self.opti.set_initial(self.X[3,:], 3.0)

        try:
            sol = self.opti.solve()
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)
            return X_opt, U_opt
        except Exception as e:
            print("Return status (if available):", getattr(self.opti, 'return_status', 'N/A'))
            self.opti.debug.show_infeasibilities(1e-7)
            print("==== END DEBUGGING ====")
            return None, None
def run_simulation():
    dyn  = CarDynamicsComplex()
    mpc  = ExactMPC(dyn)

    x    = np.array([0.0, 1.95, 0.0, 3.0, 0.0, 0.0])   
    Nsim = 1000
    hist_x, hist_u, t_solve = [x.copy()], [], []

    warm_u = None; warm_X = None
    print(f"{'Step':<6} {'x [m]':<8} {'y [m]':<8} {'θ [rad]':<10} {'v [m/s]':<8} {'δ [rad]':<10} {'τ [N]':<10} {'Time [ms]':<10}")
    for k in range(Nsim):
        start_time= time.time()
        Xopt, Uopt = mpc.solve(x, warm_X, warm_u)
        t_solve.append((time.time()-start_time)*1e3)   

        u = Uopt[:,0]
        print(f"{k:<4} {x[0]:6.2f} {x[1]:6.2f} {x[2]:7.3f} {x[3]:6.2f} {x[4]:6.2f} {x[5]:6.2f} |"
              f" {u[0]:5.2f} {u[1]:5.2f}  {t_solve[-1]:7.1f}")

        x = dyn.step(x, u)
        hist_x.append(x.copy()); hist_u.append(u.copy())
        warm_u, warm_X = Uopt, Xopt
        if x[0] >= mpc.x_ref: break
    
    hist_x = np.array(hist_x); hist_u = np.array(hist_u)
    t = np.arange(len(hist_x)) * dyn.dt

    plt.figure(figsize=(15, 12))
    
    plt.subplot(331); plt.plot(hist_x[:,0], hist_x[:,1], 'b')
    SAFE_DIST = 5.67
    ELLIPSE_X_SCALE = 3.0

    for obs_x, obs_y in mpc.obstacles:
        width  = 2 * ELLIPSE_X_SCALE * np.sqrt(SAFE_DIST)
        height = 2 * np.sqrt(SAFE_DIST)
        ell = patches.Ellipse((obs_x, obs_y), width, height,
                              angle=0, alpha=0.3, color='red')
        plt.gca().add_patch(ell)
    plt.axvline(mpc.x_ref, color='r', ls='--'); plt.axhline(mpc.y_ref, color='g', ls='--')
    plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('Position States (x,y)'); plt.axis('equal'); plt.grid()

    # ψ 
    plt.subplot(332); plt.plot(t, hist_x[:,2]); 
    plt.xlabel('Time [s]'); plt.ylabel('ψ [rad]'); plt.title('Heading Angle (ψ)'); plt.grid()
    
    # vx 
    plt.subplot(333); plt.plot(t, hist_x[:,3]); 
    plt.xlabel('Time [s]'); plt.ylabel('vx [m/s]'); plt.title('X-Velocity (vx)'); plt.grid()
    
    # vy 
    plt.subplot(334); plt.plot(t, hist_x[:,4]); 
    plt.xlabel('Time [s]'); plt.ylabel('vy [m/s]'); plt.title('Y-Velocity (vy)'); plt.grid()
    
    # ω 
    plt.subplot(335); plt.plot(t, hist_x[:,5]); 
    plt.xlabel('Time [s]'); plt.ylabel('ω [rad/s]'); plt.title('Yaw Rate (ω)'); plt.grid()
    
    # δ (control input 1)
    plt.subplot(336); plt.plot(t[:-1], hist_u[:,0]); 
    plt.xlabel('Time [s]'); plt.ylabel('δ [rad]'); plt.title('Control Input: Steering (δ)'); plt.grid()
    
    # τ (control input 2)
    plt.subplot(337); plt.plot(t[:-1], hist_u[:,1]); 
    plt.xlabel('Time [s]'); plt.ylabel('τ [Nm]'); plt.title('Control Input: Torque (τ)'); plt.grid()
    
    plt.subplot(338); plt.plot(t_solve); 
    plt.xlabel('Iteration'); plt.ylabel('Time [ms]'); plt.title('Performance: Solve Time'); plt.grid()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    run_simulation()
