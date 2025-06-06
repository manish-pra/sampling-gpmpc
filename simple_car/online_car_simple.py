import numpy as np
import time
import casadi as ca
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
np.random.seed(5)  
initial = 0

class RandomFourierFeatures:
    """
    Usage:  
    3 RandomFourierFeatures instances (one for each output dimension)
    init() -> draw omega matrix (num_features, input_dim) and b (num_features)
    features() -> enter input matrix to get feature matrix (number of input points x num_features) 
    features_casadi() -> enter one input -> feature vector in casadi
    """
    def __init__(self, input_dim, num_features, lengthscale=10.0, variance=1.0):
        self.input_dim = input_dim
        self.num_features = num_features
        self.lengthscale = lengthscale
        self.variance = variance
        
        self.omega = np.random.normal(0, 1/lengthscale, (num_features, input_dim))
        self.b = np.random.uniform(0, 2*np.pi, num_features)
    
    def features(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proj = X @ self.omega.T + self.b
        return np.sqrt(2 * self.variance / self.num_features) * np.cos(proj)
    
    def features_casadi(self, x):
        features = []
        for i in range(self.num_features):
            omega_i = self.omega[i, :]
            b_i = self.b[i]
            proj = ca.dot(omega_i, x) + b_i
            features.append(ca.cos(proj))
        
        return ca.sqrt(2 * self.variance / self.num_features) * ca.vertcat(*features)

class CarDynamicsGP:
    """
    Usage:
    One CarDynamicsGP instance
    generate_training_data() -> generate training data by sampling from true dynamics + noise, arbitrary initial state and control input
    true_dynamics() -> true dynamics step given state and control input
    train_gp() -> train gp by using the training data
    sample_weights() -> sample weights from the posterior distribution
    _sequential_update() -> updates the posterior distribution by using the new measurement at each MPC step for one output dimension and returns the new posterior mean and new cholesky factor
    online_update() -> uses the sequential update to update the posterior distribution for each output dimension and updates weight_samples
    """
    def __init__(self, num_features=[30,30,40]):
        self.lf = 1.105  
        self.lr = 1.738  
        self.dt = 0.06   
        
        self.num_features = num_features
        self.noise_var = 1e-4
        
        self.rff_models = [RandomFourierFeatures(3, num_features[0]), RandomFourierFeatures(3, num_features[1]), RandomFourierFeatures(2, num_features[2])] 
        self.weights_posterior_mean = [None] * 3
        self.weights_posterior_cov = [None] * 3
    
    # use formula from paper to perform a true dynamics step
    def true_dynamics(self, state, control):
        x_p, y_p, theta, v = state
        delta, a = control
        
        zeta = np.arctan(self.lr / (self.lf + self.lr) * np.tan(delta))
        
        f_xu = np.array([
            x_p,                    
            y_p,                   
            theta,                 
            v + a * self.dt        
        ])
        
        g_xu = np.array([
            v * np.cos(theta + zeta) * self.dt,  
            v * np.sin(theta + zeta) * self.dt, 
            v * np.sin(zeta) / self.lr * self.dt
        ])
        
        next_state = f_xu.copy()
        next_state[:3] += g_xu  
        
        return next_state
    
    # generate num_trajectories trajectories with initial state/input distribution taken from paper   
    # g_noisy = next_state_true - f_known + noise 
    def generate_training_data(self, num_trajectories=1, trajectory_length=100):
        X_train_xy = []
        X_train_theta = []
        Y_train = []
        
        for _ in range(num_trajectories):
            state = np.array([
                np.random.uniform(-2.14, 70),    
                np.random.uniform(0, 6),     
                np.random.uniform(-1.14, 1.14),
                np.random.uniform(-1, 15)      
            ])
            
            for _ in range(trajectory_length):
                control = np.array([
                    np.random.uniform(-0.6, 0.6),  
                    np.random.uniform(-2, 2)      
                ])
                
                next_state_true = self.true_dynamics(state, control)
                f_known = np.array([state[0], state[1], state[2], state[3] + control[1] * self.dt])
                g_true = next_state_true[:3] - f_known[:3]
                gp_input_xy = np.array([state[2], state[3], control[0]])
                gp_input_theta = np.array([state[3], control[0]])
                g_noisy = g_true + np.random.normal(0, np.sqrt(self.noise_var), 3)
                X_train_xy.append(gp_input_xy)
                X_train_theta.append(gp_input_theta)
                Y_train.append(g_noisy)
                
                state = next_state_true
        print("reached" )
        return (np.array(X_train_xy),
            np.array(X_train_theta),
            np.array(Y_train))
    # for each output dim create feature matrix using RFF instance-> create posterior mean and covariance matrix
    def train_gp(self, X_train_xy, X_train_theta, Y_train):
        N = X_train_xy.shape[0]
        
        for dim in range(3):
            X_train = X_train_xy if dim < 2 else X_train_theta
            Phi = self.rff_models[dim].features(X_train)
            A = Phi.T @ Phi + self.noise_var * np.eye(self.num_features[dim])
            A_inv = np.linalg.inv(A)
            
            self.weights_posterior_mean[dim] = A_inv @ Phi.T @ Y_train[:, dim]
            self.weights_posterior_cov[dim] = self.noise_var * A_inv
            eigvals = np.linalg.eigvalsh(self.weights_posterior_cov[dim])
            print(f"dim {dim}:  max eigenvalue = {eigvals.max():.3e}")
    
    def sample_weights(self, num_samples):
        global initial
        self.z_bank = []                
        self.L_bank = []               
        self.weight_samples = []      

        for dim in range(3):
            L = cholesky(self.weights_posterior_cov[dim], lower=True)
            self.L_bank.append(L)

        for _ in range(num_samples):
            w_n = []; z_n = []
            for dim, L in enumerate(self.L_bank):
                z = np.random.randn(self.num_features[dim])
                w = self.weights_posterior_mean[dim] + L @ z
                w_n.append(w); z_n.append(z)
            self.weight_samples.append(w_n)
            self.z_bank.append(z_n)
        if initial == 0:
            obj = np.asarray(self.weight_samples, dtype=object)      
            np.save("gp_weight_bank.npy", obj)      
            print("✓ saved weight bank to gp_weight_bank.npy") 
            initial += 1

        return self.weight_samples
    
    # cubic complexity for cholesky decomposition -> can improve but it is not a bottleneck
    def _sequential_update(self, phi, y, dim,
                           eta=1,         
                           eps=1e-12,    
                           max_jitter=1e-4): 

        L   = self.L_bank[dim]                 
        mu  = self.weights_posterior_mean[dim]
        sig2 = self.noise_var
        u2 = L.T @ phi           
        w  = L @ u2            
        S  = u2 @ u2 + sig2     

        K  = eta * w / S         
        mu = mu + K * (y - phi @ mu)

        self.weights_posterior_cov[dim] = L @ L.T - eta * np.outer(w, w) / S
        Sigma_new = self.weights_posterior_cov[dim]

        lam_min = np.min(np.linalg.eigvalsh(self.weights_posterior_cov[dim]))
        jitter  = eps * np.trace(self.weights_posterior_cov[dim])
        while lam_min < 0.0:
            self.weights_posterior_cov[dim] += (-lam_min + jitter) * np.eye(L.shape[0])
            try:
                L_new = np.linalg.cholesky(Sigma_new)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > max_jitter:
                    raise RuntimeError(
                        "Σ not pd even after jitter escalation."
                    )
            lam_min = np.min(np.linalg.eigvalsh(Sigma_new))
        else:
            L_new = np.linalg.cholesky(Sigma_new)

        self.weights_posterior_mean[dim] = mu
        self.L_bank[dim]                 = L_new
        return L_new, mu
  





    # sequential update for all output dimensions
    def online_update(self, x_xy, x_th, y_vec):
        phi0 = self.rff_models[0].features(x_xy).ravel()   
        phi1 = self.rff_models[1].features(x_xy).ravel()  
        phi2 = self.rff_models[2].features(x_th).ravel()  

        Ls = []; mus = []
        for dim, (phi, y) in enumerate(((phi0, y_vec[0]),   
                                      (phi1, y_vec[1]),   
                                      (phi2, y_vec[2]))): 
            L_new, mu_new = self._sequential_update(phi, y, dim) 
            Ls.append(L_new); mus.append(mu_new)


        for n in range(len(self.weight_samples)):
            for dim in range(3):
                z   = self.z_bank[n][dim]           
                w   = mus[dim] + Ls[dim] @ z
                self.weight_samples[n][dim] = w

                    

class SafeCarMPC:
    """
    Usage:
    One SafeCarMPC instance
    setup_mpc() -> setup the mpc problem
    dynamics_casadi() -> dynamics in casadi for dynamics constraints
    update_parameters() -> update the parameters in the opti object by copying the weights from the car_gp instance
    solve_mpc() -> solve the mpc problem and return the optimal control and state trajectory
    """
    def __init__(self, car_gp, weight_samples, horizon=8, lambda_slack=1e4):
        self.car_gp = car_gp
        self.weight_samples = weight_samples
        self.num_samples = len(weight_samples)
        self.W_param = [
            [ None for _ in range(3) ]  
            for _ in range(self.num_samples)
        ]
        self.horizon = horizon
        self.dt = car_gp.dt
        self.lambda_slack = lambda_slack
        self.Q = np.diag([2, 36, 0.07, 0.005])
        self.R = np.diag([2, 2])
        
        self.x_ref = 70.0
        self.y_ref = 1.95
        
        self.state_bounds = {
            'x_min': 0, 'x_max': 100,
            'y_min': 0, 'y_max': 6, 
            'theta_min': -1.14, 'theta_max': 1.14,
            'v_min': -1, 'v_max': 15
        }
        self.control_bounds = {
            'delta_min': -0.6, 'delta_max': 0.6,
            'a_min': -2, 'a_max': 2
        }
        
        self.obstacles = [(20,2), (40,2)]


        
    def setup_mpc(self):
        # define variables 
        self.opti = ca.Opti()
        for n in range(self.num_samples):
            for dim in range(3):
                F = self.car_gp.num_features[dim]
                self.W_param[n][dim] = self.opti.parameter(F)
        self.X_opt = {}
        for n in range(self.num_samples):
            self.X_opt[n] = self.opti.variable(4, self.horizon + 1)
        self.U_opt = self.opti.variable(2, self.horizon)
        self.s_obs = self.opti.variable(len(self.obstacles), self.horizon + 1)

        self.opti.subject_to(ca.vec(self.s_obs) >= 0)
        self.x0_param = self.opti.parameter(4)
        # cost
        cost = 0
        for n in range(self.num_samples):
            for k in range(self.horizon):
                x_error = self.X_opt[n][:, k] - ca.vertcat(self.x_ref, self.y_ref, 0, 0)
                cost += ca.mtimes([x_error.T, self.Q, x_error])
                
                cost += ca.mtimes([self.U_opt[:, k].T, self.R, self.U_opt[:, k]])
                
        cost += self.lambda_slack * ca.sumsqr(self.s_obs)   

                
        cost /= self.num_samples  
        self.opti.minimize(cost)
        
        # constraints
        for n in range(self.num_samples):
            self.opti.subject_to(self.X_opt[n][:, 0] == self.x0_param)
            
            for k in range(self.horizon):
                x_next = self.dynamics_casadi(self.X_opt[n][:, k], self.U_opt[:, k], n)
                self.opti.subject_to(self.X_opt[n][:, k+1] == x_next)
        
        for n in range(self.num_samples):
            for k in range(self.horizon + 1):
                self.opti.subject_to(self.X_opt[n][0, k] >= self.state_bounds['x_min'])
                self.opti.subject_to(self.X_opt[n][0, k] <= self.state_bounds['x_max'])
                self.opti.subject_to(self.X_opt[n][1, k] >= self.state_bounds['y_min'])
                self.opti.subject_to(self.X_opt[n][1, k] <= self.state_bounds['y_max'])
                self.opti.subject_to(self.X_opt[n][2, k] >= self.state_bounds['theta_min'])
                self.opti.subject_to(self.X_opt[n][2, k] <= self.state_bounds['theta_max'])
                self.opti.subject_to(self.X_opt[n][3, k] >= self.state_bounds['v_min'])
                self.opti.subject_to(self.X_opt[n][3, k] <= self.state_bounds['v_max'])
                
                for obs_i, (obs_x, obs_y) in enumerate(self.obstacles):
                    dist_sq = ((self.X_opt[n][0, k] - obs_x)**2) / 9 \
                            +  (self.X_opt[n][1, k] - obs_y)**2
                    self.opti.subject_to(dist_sq + self.s_obs[obs_i, k] >= 5.67)
            for k in range(self.horizon):
                self.opti.subject_to(self.U_opt[0, k] >= self.control_bounds['delta_min'])
                self.opti.subject_to(self.U_opt[0, k] <= self.control_bounds['delta_max'])
                self.opti.subject_to(self.U_opt[1, k] >= self.control_bounds['a_min'])
                self.opti.subject_to(self.U_opt[1, k] <= self.control_bounds['a_max'])
        
        # solver
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 1000
        }
        self.opti.solver('ipopt', opts)
 
    # used in constraints
    def dynamics_casadi(self, state, control, sample_idx):
        x_p, y_p, theta, v = state[0], state[1], state[2], state[3]
        delta, a = control[0], control[1]
        
        f_known = ca.vertcat(x_p, y_p, theta, v + a * self.dt)
        
        gp_input_xy = ca.vertcat(theta, v, delta)
        gp_input_theta = ca.vertcat(v, delta) 
        g_pred_list = []
        # for x and y (the input dimension is 3)
        for dim in range(2):
            features = self.car_gp.rff_models[dim].features_casadi(gp_input_xy)
            weights = self.W_param[sample_idx][dim]
            g_pred_dim = ca.dot(weights, features)
            g_pred_list.append(g_pred_dim)
        
        # for theta (the input dimension is 2)
        features = self.car_gp.rff_models[2].features_casadi(gp_input_theta)
        weights = self.W_param[sample_idx][2]
        g_pred_dim = ca.dot(weights,features)
        g_pred_list.append(g_pred_dim)
        g_pred = ca.vertcat(*g_pred_list)
        
        next_state = ca.vertcat(
            f_known[0] + g_pred[0],
            f_known[1] + g_pred[1], 
            f_known[2] + g_pred[2],
            f_known[3]
        )
        
        return next_state
    # at each step this function is called to update the parameters in the opti object: the online update is done in the car_gp class and 
    # the weights are copied into the opti object
    def update_parameters(self):
        for n in range(self.num_samples):
            for dim in range(3):
                self.opti.set_value(
                    self.W_param[n][dim],
                    self.weight_samples[n][dim]
                )
    # solves mpc after initialization
    def solve_mpc(self, current_state, warm_start_u=None, x_warm = None):
        self.opti.set_value(self.x0_param, current_state)
        if warm_start_u is not None:
             u_shifted = np.hstack([warm_start_u[:, 1:], warm_start_u[:, -1:]])
             self.opti.set_initial(self.U_opt, u_shifted)
             for n in range(self.num_samples):
                 self.opti.set_initial(self.X_opt[n], x_warm[n])
                 self.opti.set_initial(self.X_opt[n][:, 0], current_state)
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U_opt)
            x_opt = {}
            for n in range(self.num_samples):
                x_opt[n] = sol.value(self.X_opt[n])
            
            return u_opt, x_opt, True
        except RuntimeError as e:
            print("==== END DEBUGGING ====")
            print("→ IPOPT raised:", e)
            print("\n==== Infeasible NLP – violated constraints ====")
            try:
                self.opti.debug.show_infeasibilities(1e-6)
            except TypeError:
                self.opti.debug.show_infeasibilities()
            print("================================================\n")
            return None, None, False
    """
    # this function can be used in solve_mpc for efficient initialization
    def simulate_sample(self, state, control, sample_idx):
        f_known = np.array([state[0], state[1], state[2], state[3] + control[1] * self.dt])
        
        gp_input_xy = np.array([state[2], state[3], control[0]])
        gp_input_theta = np.array([state[3], control[0]])
        g_pred = np.zeros(3)
        
        for dim in range(2):
            features = self.car_gp.rff_models[dim].features(gp_input_xy.reshape(1, -1))
            weights = self.weight_samples[sample_idx][dim]
            g_pred[dim] = np.dot(weights, features.flatten())
        # for theta
        features = self.car_gp.rff_models[2].features(gp_input_theta.reshape(1, -1))
        weights = self.weight_samples[sample_idx][2]
        g_pred[2] = np.dot(weights, features.flatten())
        next_state = f_known.copy()
        next_state[:3] += g_pred
        
        return next_state
    """
    # main function to run the simulation
def run_simulation():
    print("Setting up car dynamics GP...")
    
    car_gp = CarDynamicsGP(num_features=[30,30,40])
    
    print("Generating training data...")
    X_train_xy, X_train_theta, Y_train = car_gp.generate_training_data(num_trajectories=1, trajectory_length=45)
    print(f"Training data: {X_train_xy.shape[0]} points")
    
    print("Training GP...")
    car_gp.train_gp(X_train_xy, X_train_theta, Y_train)
    
    print("Sampling weight matrices...")
    num_samples = 5
    weight_samples = car_gp.sample_weights(num_samples)
    
    print("Setting up MPC...")
    mpc = SafeCarMPC(car_gp, weight_samples)
    mpc.setup_mpc()
    
    sim_steps = 1000 
    initial_state = np.array([0.0, 1.95, 0.0, 3.0])  
    
    states_history = [initial_state]
    controls_history = []
    solve_times = []
    
    current_state = initial_state.copy()
    warm_start_u = None
    x_warm = None
    print("Running MPC simulation...")
    print("=" * 80)
    print(f"{'Step':<6} {'x [m]':<8} {'y [m]':<8} {'θ [rad]':<10} {'v [m/s]':<8} {'δ [rad]':<10} {'a [m/s²]':<10} {'Time [ms]':<10}")
    print("=" * 80)
    
    for step in range(sim_steps):
        start_time = time.time()
        # push new weights into casadi
        mpc.update_parameters()   
        # solve mpc
        u_opt, x_opt, success = mpc.solve_mpc(current_state, warm_start_u, x_warm)
        solve_time = time.time() - start_time
        solve_times.append(solve_time)
        
        if not success:
            print("MPC failed, stopping simulation")
            break
        
        u_applied = u_opt[:, 0]
        controls_history.append(u_applied)
         
        print(f"{step+1:<6} {current_state[0]:<8.2f} {current_state[1]:<8.2f} {current_state[2]:<10.3f} {current_state[3]:<8.2f} {u_applied[0]:<10.3f} {u_applied[1]:<10.3f} {solve_time*1000:<10.1f}")
        
        next_state = car_gp.true_dynamics(current_state, u_applied)

        # online posterior update
        gp_input_xy    = np.array([current_state[2], current_state[3], u_applied[0]])
        gp_input_theta = np.array([current_state[3], u_applied[0]])
        f_known        = current_state[:3]
        g_true         = next_state[:3] - f_known
        g_noisy        = g_true + np.random.normal(0,
                                           np.sqrt(car_gp.noise_var),
                                           size=3)
        car_gp.online_update(gp_input_xy, gp_input_theta, g_noisy)


        states_history.append(next_state)
        current_state = next_state
        
        warm_start_u = u_opt
        x_warm = x_opt
        if current_state[0] >= mpc.x_ref:
            print("=" * 80)
            print("Reached target!")
            break
    print("=" * 80)
    
    states_history = np.array(states_history)
    controls_history = np.array(controls_history)
    
    print(f"Average solve time: {np.mean(solve_times):.3f}s ({np.mean(solve_times)*1000:.1f} ms)")
    print(f"Max solve time: {np.max(solve_times):.3f}s ({np.max(solve_times)*1000:.1f} ms)")
    print(f"Min solve time: {np.min(solve_times):.3f}s ({np.min(solve_times)*1000:.1f} ms)")
    
    print("\n" + "=" * 50)
    print("CONTROL INPUT SUMMARY")
    print("=" * 50)
    print(f"Steering angle δ:")
    print(f"  Mean: {np.mean(controls_history[:, 0]):.3f} rad ({np.degrees(np.mean(controls_history[:, 0])):.1f}°)")
    print(f"  Max:  {np.max(controls_history[:, 0]):.3f} rad ({np.degrees(np.max(controls_history[:, 0])):.1f}°)")
    print(f"  Min:  {np.min(controls_history[:, 0]):.3f} rad ({np.degrees(np.min(controls_history[:, 0])):.1f}°)")
    print(f"  Std:  {np.std(controls_history[:, 0]):.3f} rad")
    
    print(f"\nAcceleration a:")
    print(f"  Mean: {np.mean(controls_history[:, 1]):.3f} m/s²")
    print(f"  Max:  {np.max(controls_history[:, 1]):.3f} m/s²")
    print(f"  Min:  {np.min(controls_history[:, 1]):.3f} m/s²")
    print(f"  Std:  {np.std(controls_history[:, 1]):.3f} m/s²")

    # plot the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
        
    import matplotlib.patches as patches        
    SAFE_DIST = 5.67
    ELLIPSE_X_SCALE = 3.0                      


    plt.plot(states_history[:, 0], states_history[:, 1], 'b-', lw=2, label='Actual trajectory')

    for obs_x, obs_y in mpc.obstacles:
        width  = 2 * ELLIPSE_X_SCALE * np.sqrt(SAFE_DIST)
        height = 2 * np.sqrt(SAFE_DIST)
        ell = patches.Ellipse((obs_x, obs_y), width, height,
                              angle=0, alpha=0.3, color='red')
        plt.gca().add_patch(ell) 
    plt.axhline(y=mpc.y_ref, color='g', linestyle='--', label='Reference lane')
    plt.axvline(x=mpc.x_ref, color='r', linestyle='--', label='Target')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Vehicle Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    time_vec = np.arange(len(states_history)) * car_gp.dt
    
    plt.subplot(2, 3, 2)
    plt.plot(time_vec, states_history[:, 2], 'b-')
    plt.xlabel('Time [s]')
    plt.ylabel('Heading angle [rad]')
    plt.title('Heading Angle')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(time_vec, states_history[:, 3], 'b-')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity')
    plt.grid(True)
    
    time_vec_u = np.arange(len(controls_history)) * car_gp.dt
    
    plt.subplot(2, 3, 4)
    plt.plot(time_vec_u, controls_history[:, 0], 'r-', linewidth=2)
    plt.axhline(y=mpc.control_bounds['delta_max'], color='r', linestyle='--', alpha=0.5, label='Bounds')
    plt.axhline(y=mpc.control_bounds['delta_min'], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Steering angle [rad]')
    plt.title('Applied Steering Input')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(time_vec_u, controls_history[:, 1], 'r-', linewidth=2)
    plt.axhline(y=mpc.control_bounds['a_max'], color='r', linestyle='--', alpha=0.5, label='Bounds')
    plt.axhline(y=mpc.control_bounds['a_min'], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Applied Acceleration Input')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(solve_times, 'g-')
    plt.axhline(y=car_gp.dt, color='r', linestyle='--', label=f'Real-time limit ({car_gp.dt*1000:.0f} ms)')
    plt.xlabel('MPC iteration')
    plt.ylabel('Solve time [s]')
    plt.title('Computation Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return states_history, controls_history, solve_times

if __name__ == "__main__":
    states, controls, times = run_simulation()
    
    print(f"\nSimulation completed!")
    print(f"Final position: x={states[-1, 0]:.2f}, y={states[-1, 1]:.2f}")
    print(f"Average computation time: {np.mean(times)*1000:.1f} ms")
    
