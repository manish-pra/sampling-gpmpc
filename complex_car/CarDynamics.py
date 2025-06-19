import numpy as np, casadi as ca
# defines the model
class CarDynamicsComplex:
    def __init__(self, dt=0.001, lf=0.052, lr=0.038, Cf=1.5, Bf=5.2, Cr=1.45, Br=8.5, Iz=5.05e-4, m=0.181, Df=0.65, Dr=1.0, Cm1=0.98028992, Cm2=0.01814131, wR=0.0175):
        self.dt, self.lf, self.lr = dt, lf, lr
        self.Cf, self.Cr = Cf, Cr 
        self.Bf, self.Br = Bf, Br
        self.Iz, self.m = Iz, m
        self.Df, self.Dr = Df, Dr
        self.Cm1, self.Cm2, self.wR = Cm1, Cm2, wR
    # helper functions
    def _a_f(self, vx, vy, omega, delta, backend):
        return backend.arctan2(vy + self.lf*omega, vx) - delta

    def _a_r(self, vx, vy, omega, backend):
        return backend.arctan2(vy - self.lr*omega, vx)

    def _Ff_no_param(self, a_f, backend):
        return backend.sin(self.Cf * backend.arctan(self.Bf * a_f))

    def _Fr_no_param(self, a_r, backend):
        return backend.sin(self.Cr * backend.arctan(self.Br * a_r))

    def _Fx(self, vx, tau):
        return (self.Cm1 - self.Cm2 * vx) * tau / self.wR
    
    # known part of right hand side (not considering the last state)
    def f_known(self, x, u, backend):
        if backend == np:
            _ , _, psi, vx, vy, omega = x
            _, tau = u
        else:
            psi, vx, vy, omega =  x[2], x[3], x[4], x[5]
            tau = u[1]

        Fx = self._Fx(vx, tau)
        f_known_xpos = vx * backend.cos(psi) - vy * backend.sin(psi)
        f_known_ypos = vx * backend.sin(psi) + vy * backend.cos(psi)
        f_known_psi = omega
        f_known_vx = 1/self.m * (Fx + self.m * vy * omega)   
        f_known_vy = 1/self.m * (-self.m * vx * omega)
        f_known_omega = 0.0
        f_list = [f_known_xpos, f_known_ypos, f_known_psi, f_known_vx, f_known_vy, f_known_omega]
        if backend == np:
            return np.array(f_list)
        else:
            return ca.vertcat(*f_list)
    
    # features: only vx and vy and omega have non-zero feature functions
    def features(self, x, u, backend):
        delta = u[0]
        vx, vy, omega = x[3], x[4], x[5]
        a_f = self._a_f(vx, vy, omega, delta, backend)
        a_r = self._a_r(vx, vy, omega, backend)
        Ff_no_param = self._Ff_no_param(a_f, backend)   
        Fr_no_param = self._Fr_no_param(a_r, backend)
        if backend == np:
            phi_vx = np.array([(-np.sin(delta)/self.m) * Ff_no_param,
                           0.0])
            phi_vy = np.array([( np.cos(delta)/self.m) * Ff_no_param,
                            (1.0/self.m) * Fr_no_param])
            phi_omega = np.array([( self.lf*np.cos(delta)/self.Iz) * Ff_no_param,
                                (-self.lr/self.Iz) * Fr_no_param])
            return np.vstack([[0.0,0.0], [0.0,0.0], [0.0,0.0], phi_vx, phi_vy, phi_omega])
        else:
            zeros3 = ca.DM.zeros(3,2)
            phi_vx = ca.hcat([(-backend.sin(delta)/self.m) * Ff_no_param,
                           0.0])
            phi_vy = ca.hcat([( backend.cos(delta)/self.m) * Ff_no_param,
                            (1.0/self.m) * Fr_no_param])
            phi_omega = ca.hcat([( self.lf*backend.cos(delta)/self.Iz) * Ff_no_param,
                                (-self.lr/self.Iz) * Fr_no_param])
            return ca.vertcat(*[zeros3, phi_vx, phi_vy, phi_omega])
    # advances state 
    def step(self, x, u, Df, Dr, backend):
        params = None
        if backend == np:
            params = np.array([Df, Dr])
        else: 
            params = ca.vertcat(Df, Dr)
        return x + self.dt * (self.f_known(x, u, backend) + self.features(x,u, backend) @ params)
    
class StraightTrack:
    center_y = 1.95

    @staticmethod
    def contouring_error(y):      
        return y - StraightTrack.center_y

    @staticmethod
    def lag_error(theta, x):      
        return theta - x
