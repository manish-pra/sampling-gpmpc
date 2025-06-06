import numpy as np, time, casadi as ca, matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(42)

class CarDynamics:
    def __init__(self, dt=0.06, lf=1.105, lr=1.738):
        self.dt, self.lf, self.lr = dt, lf, lr

    def step(self, x, u):
        xpos, ypos, th, v = x
        delta, a          = u
        zeta = np.arctan(self.lr / (self.lf + self.lr) * np.tan(delta))

        x_next = np.array([
            xpos + v * np.cos(th + zeta) * self.dt,
            ypos + v * np.sin(th + zeta) * self.dt,
            th   + v * np.sin(zeta)/self.lr * self.dt,
            v    + a * self.dt
        ])
        return x_next

    def casadi_step(self, x, u):
        xpos, ypos, th, v = x[0], x[1], x[2], x[3]
        delta, a          = u[0], u[1]
        zeta = ca.atan(self.lr / (self.lf + self.lr) * ca.tan(delta))

        x_next = ca.vertcat(
            xpos + v * ca.cos(th + zeta) * self.dt,
            ypos + v * ca.sin(th + zeta) * self.dt,
            th   + v * ca.sin(zeta)/self.lr * self.dt,
            v    + a * self.dt
        )
        return x_next

class ExactMPC:
    def __init__(self, model, horizon=20, lambda_slack=1000):
        self.mdl, self.N, self.dt = model, horizon, model.dt
        self.lambda_slack = lambda_slack

        self.Q = np.diag([ 2, 36, 0.07, 0.005])
        self.R = np.diag([ 2,  2])

        self.x_ref, self.y_ref = 70.0, 1.95
        self.bounds_state = dict(x=(0,100), y=(0,6),
                                 th=(-1.14,1.14), v=(-1,15))
        self.bounds_ctrl  = dict(delta=(-0.6,0.6), a=(-2,2))
        self.obstacles = [(20,2), (40,2)]          

        self._build_optimizer()

    def _build_optimizer(self):
        self.opti = ca.Opti()

        self.X = self.opti.variable(4, self.N+1)
        self.U = self.opti.variable(2, self.N)
        self.s_obs = self.opti.variable(len(self.obstacles), self.N+1)  

        self.x0 = self.opti.parameter(4)

        cost = 0
        for k in range(self.N):
            e = self.X[:,k] - ca.vertcat(self.x_ref, self.y_ref, 0, 0)
            cost += ca.mtimes([e.T, self.Q, e])
            cost += ca.mtimes([self.U[:,k].T, self.R, self.U[:,k]])
        cost += self.lambda_slack * ca.sumsqr(self.s_obs)
        self.opti.minimize(cost)

        self.opti.subject_to(self.X[:,0] == self.x0)
        for k in range(self.N):
            x_next = self.mdl.casadi_step(self.X[:,k], self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == x_next)

        bx, by = self.bounds_state['x'], self.bounds_state['y']
        bth, bv = self.bounds_state['th'], self.bounds_state['v']
        bδ, ba  = self.bounds_ctrl['delta'], self.bounds_ctrl['a']

        for k in range(self.N+1):
            self.opti.subject_to(self.X[0,k] >= bx[0]); self.opti.subject_to(self.X[0,k] <= bx[1])
            self.opti.subject_to(self.X[1,k] >= by[0]); self.opti.subject_to(self.X[1,k] <= by[1])
            self.opti.subject_to(self.X[2,k] >= bth[0]); self.opti.subject_to(self.X[2,k] <= bth[1])
            self.opti.subject_to(self.X[3,k] >= bv[0]);  self.opti.subject_to(self.X[3,k] <= bv[1])
            for j,(ox,oy) in enumerate(self.obstacles):
                dist = ((self.X[0,k]-ox)**2)/9 + (self.X[1,k]-oy)**2
                self.opti.subject_to(dist + self.s_obs[j,k] >= 5.67)
        for k in range(self.N):
            self.opti.subject_to(self.U[0,k] >= bδ[0]); self.opti.subject_to(self.U[0,k] <= bδ[1])
            self.opti.subject_to(self.U[1,k] >= ba[0]); self.opti.subject_to(self.U[1,k] <= ba[1])

        self.opti.subject_to(ca.vec(self.s_obs) >= 0)
        opts = {'ipopt.print_level':0, 'print_time':0}
        self.opti.solver('ipopt', opts)

    def solve(self, x_init, warm_u=None, warm_x=None):
        self.opti.set_value(self.x0, x_init)
        if warm_u is not None:
            u_init = np.hstack([warm_u[:,1:], warm_u[:,-1:]])
            self.opti.set_initial(self.U, u_init)
            self.opti.set_initial(self.X, warm_x)     

        sol = self.opti.solve()
        U_opt = sol.value(self.U)
        X_opt = sol.value(self.X)
        return U_opt, X_opt

def run():
    dyn  = CarDynamics()
    mpc  = ExactMPC(dyn)

    x    = np.array([0.0, 1.95, 0.0, 3.0])   
    Nsim = 1000
    hist_x, hist_u, t_solve = [x.copy()], [], []

    warm_u = None; warm_X = None
    print(f"{'k':<4} {'x':>6} {'y':>6} {'th':>7} {'v':>6} | δ  a  solve[ms]")
    for k in range(Nsim):
        tic = time.time()
        Uopt, Xopt = mpc.solve(x, warm_u, warm_X)
        t_solve.append((time.time()-tic)*1e3)   

        u = Uopt[:,0]
        print(f"{k:<4} {x[0]:6.2f} {x[1]:6.2f} {x[2]:7.3f} {x[3]:6.2f} |"
              f" {u[0]:5.2f} {u[1]:5.2f}  {t_solve[-1]:7.1f}")

        x = dyn.step(x, u)
        hist_x.append(x.copy()); hist_u.append(u.copy())

        warm_u, warm_X = Uopt, Xopt     
        if x[0] >= mpc.x_ref: break

    hist_x = np.array(hist_x); hist_u = np.array(hist_u)
    t = np.arange(len(hist_x)) * dyn.dt

    plt.figure(figsize=(14,8))
    plt.subplot(231); plt.plot(hist_x[:,0], hist_x[:,1],'b')
    SAFE_DIST = 5.67
    ELLIPSE_X_SCALE = 3.0

    for obs_x, obs_y in mpc.obstacles:
        width  = 2 * ELLIPSE_X_SCALE * np.sqrt(SAFE_DIST)
        height = 2 * np.sqrt(SAFE_DIST)
        ell = patches.Ellipse((obs_x, obs_y), width, height,
                              angle=0, alpha=0.3, color='red')
        plt.gca().add_patch(ell)

    plt.axvline(mpc.x_ref,color='r',ls='--'); plt.axhline(mpc.y_ref,color='g',ls='--')
    plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('trajectory'); plt.axis('equal'); plt.grid()

    # heading
    plt.subplot(232); plt.plot(t, hist_x[:,2]); plt.title('heading'); plt.grid()
    # speed
    plt.subplot(233); plt.plot(t, hist_x[:,3]); plt.title('speed'); plt.grid()
    # steering
    plt.subplot(234); plt.plot(t[:-1], hist_u[:,0]); plt.title('δ'); plt.grid()
    # throttle
    plt.subplot(235); plt.plot(t[:-1], hist_u[:,1]); plt.title('a'); plt.grid()
    # solve time
    plt.subplot(236); plt.plot(t_solve); plt.title('solve time [ms]'); plt.grid()

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    run()

