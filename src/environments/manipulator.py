import torch
import numpy as np
import scipy.linalg
import scipy.signal
import casadi as ca
from urdf2casadi import urdfparser as u2c
import pybullet as p
import pybullet_data

class Manipulator(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.pad_g = [0, 1, 2, 3]  # 0, self.g_nx + self.g_nu :
        self.datax_idx = self.params["env"]["datax_idx"]
        self.datau_idx = self.params["env"]["datau_idx"]
        self.urdf_path = f"{self.params['env']['urdf_model']}.urdf"
        urdf_prefix = pybullet_data.getDataPath()
        self.parser = u2c.URDFparser()
        self.parser.from_file(urdf_prefix + '/' +self.urdf_path)
        self.base_link = self.params["env"]["base_link"]
        self.ee_link = self.params["env"]["ee_link"]
        theta = ca.SX.sym("theta", self.nu)
        theta_dot = ca.SX.sym("theta_dot", self.nu)
        tau = ca.SX.sym("tau",self.nu)
        self.theta_ddot_expr = self.parser.get_forward_dynamics_aba(self.base_link, self.ee_link)(theta, theta_dot, tau)
        self.forward_dyn = ca.Function("forward_dynamics", [theta, theta_dot, tau], [self.theta_ddot_expr])
        self.theta_ddot_fric_expr = self.parser.get_forward_dynamics_friction_aba(self.base_link, self.ee_link)(theta, theta_dot, tau)
        self.forward_dyn_fric = ca.Function("forward_dynamics", [theta, theta_dot, tau], [self.theta_ddot_fric_expr])
        self.fk = self.parser.get_forward_kinematics(self.base_link, self.ee_link)["T_fk"]
        self.tr_weights = None

        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
            self.torch_device = torch.device("cuda")
            torch.set_default_device(self.torch_device)
        else:
            self.use_cuda = False
            self.torch_device = torch.device("cpu")
            torch.set_default_device(self.torch_device)

        self.B_d = torch.eye(self.nx, self.g_ny, device=self.torch_device)

    def setup_manipulator_visu(self):
        if self.params["visu"]["show"]:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            self.log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "output_video.mp4")
            self.robot_id = p.loadURDF(self.urdf_path, 
                                       basePosition=[0, 0, 0.0],
                                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                                       useFixedBase=True,
                                       flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

            for j in range(-1, p.getNumJoints(self.robot_id)):  # -1 is base
                p.setCollisionFilterGroupMask(self.robot_id, j, 0, 0)



            # # Draw XYZ axes at the base
            # pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            # rot_matrix = p.getMatrixFromQuaternion(orn)
            # x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
            # y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
            # z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

            # scale = 0.1
            # p.addUserDebugLine(pos, [pos[0] + scale * x_axis[0], pos[1] + scale * x_axis[1], pos[2] + scale * x_axis[2]], [1, 0, 0], 2)
            # p.addUserDebugLine(pos, [pos[0] + scale * y_axis[0], pos[1] + scale * y_axis[1], pos[2] + scale * y_axis[2]], [0, 1, 0], 2)
            # p.addUserDebugLine(pos, [pos[0] + scale * z_axis[0], pos[1] + scale * z_axis[1], pos[2] + scale * z_axis[2]], [0, 0, 1], 2)

            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=00,
                cameraPitch=-45,
                cameraTargetPosition=[0.2, 0, 0.2]
            )

            # -------------------------------
            # Get revolute joint indices
            # -------------------------------
            self.joint_indices = [
                i for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE
            ]
            print("Revolute joints:", self.joint_indices)

            initial_joint_positions = self.params["env"]["start"]  # example for 7-DOF Panda
            for i, joint_index in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_index, initial_joint_positions[i])

            # -------------------------------
            # Disable default motor control
            # -------------------------------
            for j in self.joint_indices:
                p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

            # -------------------------------
            # Animate joints using sin(t)
            # -------------------------------
            p.setTimeStep(self.params["optimizer"]["dt"])
            p.setPhysicsEngineParameter(fixedTimeStep=self.params["optimizer"]["dt"])
            p.setGravity(0, 0, -9.81)
            p.setRealTimeSimulation(0)

            # Get world pose of base_link in PyBullet
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            # T_world_base = your_quat_pose_to_SE3(base_pos, base_orn)

            # T_base_ee = self.fk(q)
            # T_world_ee = T_world_base @ T_base_ee
            # ee_pos = T_world_ee[:3, 3]

            a=1

    def apply_action(self, state, action):
        target_positions = [state[i].item() for i in range(self.nu)]
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, target_positions[i])

        p.setJointMotorControlArray(self.robot_id,
            self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions= target_positions,
            forces= [50.0 for _ in range(self.nu)])  # Set target position and force for each joint
        # p.setJointMotorControlArray(self.robot_id,
        #             self.joint_indices,
        #             controlMode=p.TORQUE_CONTROL,
        #             forces= [action[0].item(), action[1].item()]) 
        # p.setJointMotorControlArray(self.robot_id,
        #         self.joint_indices,
        #         controlMode=p.VELOCITY_CONTROL,
        #         targetVelocities= [state[2].item(), state[3].item()],
        #         forces= [5.0, 5.0]) 
        # for j in range(2):
        #     # p.setJointMotorControl2(
        #     #     self.robot_id,
        #     #     j,
        #     #     controlMode=p.POSITION_CONTROL,
        #     #     targetPosition=state[j].item(),
        #     #     force=5.0,
        #     # )

        #     p.setJointMotorControl2(
        #         self.robot_id,
        #         j,
        #         controlMode=p.TORQUE_CONTROL,
        #         # targetPosition=qj,
        #         force=action[j].item()*10,
        #     )
        ######################Simulation Step######################
        # p.stepSimulation()
        joint_positions = []
        for j in self.joint_indices:
            pos, _, _, _ = p.getJointState(self.robot_id, j)
            joint_positions.append(pos)
        joint_positions = np.array(joint_positions)
        # past_joint_positions = joint_positions + 20
        target_positions = np.array(target_positions)
        # while np.linalg.norm(joint_positions - target_positions) > 0.01 and np.linalg.norm(joint_positions - past_joint_positions) > 0.001:
        #     print("Waiting for joint positions to stabilize...")
        #     p.stepSimulation()
        #     past_joint_positions = joint_positions.copy()
        #     joint_positions = []
        #     for j in self.joint_indices:
        #         pos, _, _, _ = p.getJointState(self.robot_id, j)
        #         joint_positions.append(pos)
        #     joint_positions = np.array(joint_positions) 
        print("Target positions:", self.fk(target_positions.tolist())[:3,3])
        print("Joint positions stabilized:", self.fk(joint_positions.tolist())[:3,3])

        # a=1
    
    def stop_visu(self):
        p.stopStateLogging(self.log_id)
        p.disconnect()

    def print_3d_state(self, state):
        position = self.fk(state[:self.nu].tolist())
        print("3D Position:", position[:3,3:])

    def initial_training_data(self):
        # keep low output scale, TODO: check if variance on gradient output can be controlled Dyn_gp_task_noises
        n_data_x = self.params["env"]["n_data_x"]
        n_data_u = self.params["env"]["n_data_u"]

        if self.params["env"]["prior_dyn_meas"]:
            if self.params["env"]["randomly_sample"]:
                # randomly sample points in the state and control space
                center = torch.Tensor(self.params["env"]["start"] + [0,0,0,0,0,0,0])
                std = 1 
                Dyn_gp_X_train = torch.rand(
                    (n_data_x, self.g_nx + self.g_nu),
                    device=self.torch_device,
                )
                # generate random controls
                Dyn_gp_X_train = center + std * torch.randn(self.params["env"]["num_samples"], center.shape[0])
            else:
                # generate a grid of points in the state and control space
                mesh_list = []
                for idx in self.datax_idx:
                    mesh_list.append(torch.linspace(
                    self.params["optimizer"]["x_min"][idx],
                    self.params["optimizer"]["x_max"][idx],
                    n_data_x,
                ))
                for idx in self.datau_idx:
                    mesh_list.append(torch.linspace(
                    self.params["optimizer"]["u_min"][idx],
                    self.params["optimizer"]["u_max"][idx],
                    n_data_u,
                ))
                XU_grid = torch.meshgrid(mesh_list)

                Dyn_gp_X_train = torch.hstack(
                    [XU.reshape(-1,1) for XU in XU_grid]
                )
            # y1, y2 = self.get_prior_data(Dyn_gp_X_train)
            # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
            # add noise
            Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train) + torch.randn(
                (self.g_ny, Dyn_gp_X_train.shape[0], 1 + self.g_nx + self.g_nu),
                device=self.torch_device,
            ) * torch.sqrt(torch.tensor(self.params["agent"]["BLR"]["noise_var"], dtype=torch.float32))
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["env"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, x_hat):
        # N_data = x_hat.shape[0]
        # x_hat = x_hat.numpy()
        # dt = self.params["optimizer"]["dt"]
        # f_batch = self.forward_dyn.map(N_data)
        y_ret = torch.zeros((self.g_ny, x_hat.shape[0], 1 + self.g_nx + self.g_nu))
        g_xu = self.unknown_dyn(x_hat)



        # g_xu = np.zeros(( x_hat.shape[0],self.g_ny))
        # g_xu[:,:2] = x_hat[:,:2] + x_hat[:,2:4]*dt  
        # g_xu[:,:2] = x_hat[:,2:4] + np.array(f_batch(x_hat[:,:2].T, x_hat[:,2:4].T, x_hat[:,4:6].T).T)*dt   

        y_ret[:, :, 0] = g_xu.transpose(0, 1)
        return y_ret

    def continous_dyn(self, X1, X2, U):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        X1dot = X2.clone()
        X2dot = -g * torch.sin(X1) / l + U / l
        train_data_y = torch.hstack([X1dot.reshape(-1, 1), X2dot.reshape(-1, 1)])
        return train_data_y

    def get_true_gradient(self, x_hat):
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        ret = torch.zeros((2, x_hat.shape[0], 3))
        ret[0, :, 1] = torch.ones(x_hat.shape[0])
        ret[1, :, 0] = -g * torch.cos(x_hat[:, 0]) / l
        ret[1, :, 2] = torch.ones(x_hat.shape[0]) / l

        val = self.pendulum_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
        return torch.hstack([val[:, 0].reshape(-1, 1), ret[0, :, :]]), torch.hstack(
            [val[:, 1].reshape(-1, 1), ret[1, :, :]]
        )

    def discrete_dyn(self, xu):
        dt = self.params["optimizer"]["dt"]
        dof = self.params["agent"]["dim"]["nu"]
        state_kp1 = torch.zeros((xu.shape[0], self.nx))
        # xu = xu.numpy()
        # propagation of position
        state_kp1[:,:dof] = xu[:,:dof] + xu[:,dof:2*dof]*dt  
        theta_ddot_dynamics = torch.from_numpy(np.array(self.forward_dyn(xu[:,:dof].T.numpy(), xu[:,dof:2*dof].T.numpy(), xu[:,2*dof:].T.numpy()).T)) 
        if self.params["env"]["friction"]["use"]:
            theta_ddot_dynamics = torch.from_numpy(np.array(self.forward_dyn_fric(xu[:,:dof].T.numpy(), xu[:,dof:2*dof].T.numpy(), xu[:,2*dof:].T.numpy()).T)) 
            # idx = self.params["env"]["friction"]["dof_idx"]
            # if self.params["env"]["unknown"]["partial"]:
            #     g_xu = xu[:,self.datax_idx + [self.nx + u_idx for u_idx in self.datau_idx]]
            #     theta_ddot_dynamics[:,idx] -=  self.unknown_dyn(g_xu)
            # else:
            #     theta_ddot_dynamics[:,idx] = self.unknown_dyn(xu)
        #     # think of it as a residual model
        state_kp1[:,dof:2*dof] = xu[:,dof:2*dof] + theta_ddot_dynamics*dt
        return state_kp1

    def unknown_dyn(self, xu):
        # use the same dynamics and propagate
        
        dt = self.params["optimizer"]["dt"]
        xu = xu.numpy()
        # xu contains components relevant for friction model (not the full state)
        # residual_dyn = self.friction_model(xu[:,:self.g_nx], xu[:,self.g_nx:])
        dof = self.params["agent"]["dim"]["nu"]
        # len_fric = len(self.params["env"]["friction"]["dof_idx"])
        # fric_q = xu[:, :len_fric]
        # fric_q_dot = xu[:, len_fric:2*len_fric]
        # fric_u = xu[:, 2*len_fric:]
        # app_fric = np.hstack([fric_q, np.ones((fric_q.shape[0], dof-len_fric))])
        # app_fric_dot = np.hstack([fric_q_dot, np.ones((fric_q_dot.shape[0], dof-len_fric))])
        # app_fric_u = np.hstack([fric_u, np.ones((fric_u.shape[0], dof-len_fric))])
        
        idx_list = [comp_idx - dof for comp_idx in self.params["env"]["unknown"]["component_idx"]]
        if self.params["env"]["unknown"]["partial"]:
            theta_ddot_dynamics_fric = np.array(self.forward_dyn_fric(xu[:,:dof].T, xu[:,dof:2*dof].T, xu[:,2*dof:].T).T)
            theta_ddot_dynamics = np.array(self.forward_dyn(xu[:,:dof].T, xu[:,dof:2*dof].T, xu[:,2*dof:].T).T)
            residual_dyn = theta_ddot_dynamics_fric[:,idx_list] - theta_ddot_dynamics[:,idx_list]
        else:
            theta_ddot_dynamics_fric = np.array(self.forward_dyn(xu[:,:dof].T, xu[:,dof:2*dof].T, xu[:,2*dof:].T).T)
            residual_dyn = theta_ddot_dynamics_fric[:,idx_list]
        residual_state = residual_dyn* 1.0 # * dt

        # m = self.params["env"]["params"]["m"]
        # l = self.params["env"]["params"]["l"]
        # g = self.params["env"]["params"]["g"]
        # d = self.params["env"]["params"]["d"]
        # J = self.params["env"]["params"]["J"]
        # px_k, py_k, phi_k, vx_k, vy_k, phidot_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]], xu[:, [4]], xu[:, [5]] 
        # u1_k, u2_k = xu[:, [6]], xu[:, [7]]
        
        # px_kp1 = px_k + (vx_k * torch.cos(phi_k) - vy_k*torch.sin(phi_k))* dt
        # py_kp1 = py_k + (vx_k * torch.sin(phi_k) + vy_k*torch.cos(phi_k))* dt
        # phi_kp1 = phi_k + phidot_k * dt
        # vx_kp1 = vx_k + (vy_k*phidot_k - g * torch.sin(phi_k) + torch.cos(phi_k)*d)* dt
        # vy_kp1 = vy_k + (-vx_k*phidot_k - g * torch.cos(phi_k) + u1_k/m + u2_k/m - torch.sin(phi_k)*d)* dt
        # phidot_kp1 = phidot_k + (u1_k - u2_k)*l/J*dt
        # state_kp1 = torch.hstack([px_kp1, py_kp1, phi_kp1, vx_kp1, vy_kp1, phidot_kp1])
        return torch.from_numpy(residual_state)
    
    def get_gt_weights(self):
        dt = self.params["optimizer"]["dt"]
        tr_weight = [[1.0, dt] for _ in range(self.params["agent"]["dim"]["nx"])]
        if self.params["agent"]["run"]["true_param_as_sample"]:
            return tr_weight
        elif self.params["agent"]["g_dim"]["ny"] == self.params["agent"]["dim"]["nx"]:
            return [weight.T for weight in self.tr_weights]
        else:
            # ####NEW CODE
            # dof = self.params["agent"]["dim"]["nu"]
            # for i, idx in enumerate(self.params["env"]["friction"]["dof_idx"]):
            #     tr_weight[dof+idx] += mean[i]
            fric_idx = self.params["env"]["friction"]["dof_idx"]
            dof = self.params["agent"]["dim"]["nu"]
            dt = self.params["optimizer"]["dt"]
            for i, unknown_idx in enumerate(self.params["env"]["unknown"]["component_idx"]):
                mean_weights = dt * self.tr_weights[i]
                if self.params["env"]["unknown"]["partial"]:
                    tr_weight[unknown_idx] += mean_weights.reshape(-1).tolist()
                else:                    # if partial unknown, add the friction model
                    tr_weight[unknown_idx] = mean_weights.reshape(-1).tolist()
            # ####
            # merge the above two in with remaining dim
            # for idx in range(self.params["agent"]["g_dim"]["ny"], self.params["agent"]["dim"]["nx"]):
            #     tr_weight[idx] = self.tr_weights[idx-self.params["agent"]["g_dim"]["ny"]].T
            return tr_weight

    def get_f_known_jacobian(self, xu):
        ns = xu.shape[0]
        nH = xu.shape[2]
        # dimension is ns, ny, H, nx+nu
        df_dxu_grad = torch.zeros(
            (ns, self.nx, nH, 1 + self.nx + self.nu), device=xu.device
        )
        return df_dxu_grad

    def get_g_xu_hat(self, xu_hat):
        return xu_hat

    def LQR_controller(self):
        g = self.params["env"]["params"]["g"]
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        b = 0

        R = np.diag(np.array(self.params["optimizer"]["Qu"]))
        Qx = np.diag(np.array(self.params["optimizer"]["Qx"]))

        # Continuous-time state-space matrices
        A = np.array([[0, 1], [-g / l, 0]])  # Change sign for new coordinate system
        B = np.array([[0], [1]])

        dt = self.params["optimizer"]["dt"]
        # Discretize the system using zero-order hold (ZOH)
        system = scipy.signal.cont2discrete((A, B, np.eye(2), 0), dt, method="zoh")
        A_d, B_d, _, _ = system[:4]

        # Solve the Discrete-time Algebraic Riccati Equation (DARE)
        P = scipy.linalg.solve_discrete_are(A_d, B_d, Qx, R)

        # Compute the Discrete LQR gain K
        K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
        print(K, P)

        return K, P, A_d, B_d

    def traj_initialize(self, x_curr):
        # Store results
        x_history = []
        u_history = []
        x = x_curr
        K, P, A_d, B_d = self.LQR_controller()
        # Run simulation
        for _ in range(self.params["optimizer"]["H"]):
            u = -K @ x  # LQR control law
            x = A_d @ x + B_d @ u  # Discrete-time state update
            x_history.append(x.flatten())
            u_history.append(u.flatten())
        return x_history, u_history

    def transform_sensitivity(self, dg_dxu_grad, xu_hat):
        return dg_dxu_grad

    def propagate_true_dynamics(self, x_init, U):
        state_list = []
        state_list.append(x_init)
        for ele in range(U.shape[0]):
            state_input = (
                torch.from_numpy(np.hstack([state_list[-1], U[ele]]))
                .reshape(1, -1)
                .float()
            )
            state_kp1 = self.discrete_dyn(state_input)
            state_list.append(state_kp1.reshape(-1))
        return np.stack(state_list)
    
    def feature_rff(self, state, control, idx):
        omega = (1.0 / self.ls) * np.random.default_rng(idx).normal(size=(self.num_features, self.g_nx + self.g_nu))
        phase = np.random.default_rng(idx).uniform(0, 2 * np.pi, size=self.num_features)
        # create casadi features
        proj = omega@ca.vertcat(state,control) + phase
        return np.sqrt(2.0 / self.num_features) * ca.cos(proj)
    
    def friction_model(self, state, control):
        dof = self.params["agent"]["dim"]["nu"]
        friction_dof_idx = self.params["env"]["friction"]["dof_idx"]
        len_fric = len(friction_dof_idx)
        damping_coeff = self.params["env"]["friction"]["damping_coeff"]
        # returns damping_coeff * velocity
        if isinstance(state,np.ndarray):
            if self.params["env"]["unknown"]["partial"]:
                # numpy
                return damping_coeff * (state[:,[idx for idx in range(len_fric, 2*len_fric)]]**2)
            else:
                return damping_coeff * (state[:,self.params["env"]["unknown"]["component_idx"]]**2)
        else: # full state
            # casadi
            return damping_coeff * (state[[idx for idx in range(len_fric, 2*len_fric)]]**2)

    def theta_ddot_dynamics(self, state, control):
        # introduces friction in the dynamics
        dof = self.params["agent"]["dim"]["nu"]
        theta_ddot = self.parser.get_forward_dynamics_aba(self.base_link, self.ee_link)(state[:dof], state[dof:2*dof], control)
        if self.params["env"]["friction"]["use"]:
            theta_ddot = self.parser.get_forward_dynamics_friction_aba(self.base_link, self.ee_link)(state[:dof], state[dof:2*dof], control)
            # dof = self.params["agent"]["dim"]["nu"]
            # friction_xv_idx = self.params["env"]["friction"]["dof_idx"] + [dof+idx for idx in self.params["env"]["friction"]["dof_idx"]]
            # theta_ddot[self.params["env"]["friction"]["dof_idx"]] -= self.friction_model(state[[idx for idx in friction_xv_idx]], control[[idx for idx in self.params["env"]["friction"]["dof_idx"]]])
        return theta_ddot

    def features_partial_rff(self, state, control, idx):
        dof = self.params["agent"]["dim"]["nu"]
        if idx<dof:
            return ca.vertcat(state[idx], state[idx+dof])
        else:
            theta_ddot = self.parser.get_forward_dynamics_aba(self.base_link, self.ee_link)(state[:dof], state[dof:2*dof], control)
            if self.params["env"]["friction"]["use"]:
                if idx in self.params["env"]["unknown"]["component_idx"]:
                    additional_features = self.feature_rff(state, control, idx)
                    if self.params["env"]["unknown"]["partial"]:
                        return ca.vertcat(state[idx], theta_ddot[idx-dof], additional_features)
                    else: 
                        return additional_features
                else: # idx is not in the unknown component
                    theta_ddot = self.parser.get_forward_dynamics_friction_aba(self.base_link, self.ee_link)(state[:dof], state[dof:2*dof], control)                        
            return ca.vertcat(state[idx], theta_ddot[idx-dof])

    def features_true(self, state, control, idx):
        dof = self.params["agent"]["dim"]["nu"]
        if idx<dof:
            return ca.vertcat(state[idx], state[idx+dof])
        else:
            theta_ddot = self.theta_ddot_dynamics(state, control)
            return ca.vertcat(state[idx], theta_ddot[idx-dof])

    def BLR_features_casadi(self):
        self.num_features = self.params["env"]["num_features"]
        self.ls = self.params["agent"]["Dyn_gp_lengthscale"]["val"]


        theta = ca.SX.sym("theta", self.nu)
        theta_dot = ca.SX.sym("theta_dot", self.nu)
        tau = ca.SX.sym("tau", self.nu)

        state = ca.vertcat(theta, theta_dot)
        control = ca.vertcat(tau)

        features_name = ['f_theta1', 'f_theta2', 'f_theta3', 'f_theta4', 'f_theta5', 'f_theta6', 'f_theta7', 'f_dtheta1', 'f_dtheta2', 'f_dtheta3', 'f_dtheta4', 'f_dtheta5', 'f_dtheta6', 'f_dtheta7']

        if self.params["agent"]["run"]["true_param_as_sample"]:
            feature_function = self.features_true
        else:
            feature_function = self.features_partial_rff
        
        dof = self.params["agent"]["dim"]["nu"]
        fric_xv_idx = self.params["env"]["friction"]["dof_idx"] + [dof+idx for idx in self.params["env"]["friction"]["dof_idx"]]
        fric_u_idx = self.params["env"]["friction"]["dof_idx"]
        if self.params["env"]["unknown"]["partial"]:
            f_unknown_list = [ca.Function(features_name[idx] + "_partial", [state, control], [self.feature_rff(state, control, idx)]) for 
                 idx, feat_name in enumerate(features_name)]
        else:
            f_unknown_list = [ca.Function(features_name[idx] + "_full", [state, control], [self.feature_rff(state, control, idx)]) for 
                 idx in self.params["env"]["unknown"]["component_idx"]]

        f_list = [ca.Function(feat_name, [state, control], [feature_function(state, control, idx)]) for 
                 idx, feat_name in enumerate(features_name)]
        
        f_jac_list = [ca.Function(feat_name + "_jac", [state, control], [ca.jacobian(f(state, control), state)]) for 
                      idx, (f, feat_name) in enumerate(zip(f_list, features_name))]
        
        f_ujac_list = [ca.Function(feat_name + "_ujac", [state, control], [ca.jacobian(f(state, control), control)]) for
                        idx, (f, feat_name) in enumerate(zip(f_list, features_name))]


        batch_size = self.params["agent"]["num_dyn_samples"]
        batch_1 = 1
        batch_2 = self.params["optimizer"]["H"]
        total_samples = batch_size * batch_1 * batch_2

        f_batch_list = [f.map(total_samples, 'thread', 24) for f in f_list]
        f_jac_batch_list = [f_jac.map(total_samples, 'thread', 24) for f_jac in f_jac_list]
        f_ujac_batch_list = [f_ujac.map(total_samples, 'thread', 24) for f_ujac in f_ujac_list]

        return f_list, f_jac_list, f_ujac_list, f_batch_list, f_jac_batch_list, f_ujac_batch_list, f_unknown_list


    def BLR_features(self, X):    
        theta = X[:, [0]]
        omega = X[:, [1]]
        alpha = X[:, [2]]
        # theta, vel, alpha

        f1 = np.hstack([theta,omega])
        f2 = np.hstack([omega, np.sin(theta),alpha])
        return f1, f2
        # return np.vstack([f1, f2])

    def feature_px(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(px, vx * ca.cos(phi), vy * ca.sin(phi))

    def feature_py(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(py, vx * ca.sin(phi), vy * ca.cos(phi))

    def feature_phi(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(phi, phidot)

    def feature_vx(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(vx, vy * phidot, ca.sin(phi), ca.cos(phi))

    def feature_vy(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        u1, u2 = control[0], control[1]
        return ca.vertcat(vy, vx * phidot, ca.cos(phi), ca.sin(phi), u1, u2)

    def feature_phidot(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        u1, u2 = control[0], control[1]
        return ca.vertcat(phidot, u1, u2)

    def BLR_features_test(self, X):
        if X.ndim == 2:
            # X = X.reshape(X.shape[0], 1, X.shape[1], 1)
            X = X[:, np.newaxis, np.newaxis,:]
        # X shape: (batch_size, 2, horizon, 8)
        px = X[:, 0:1, :, 0:1]
        py = X[:, 0:1, :, 1:2]
        phi = X[:, 0:1, :, 2:3]
        vx = X[:, 0:1, :, 3:4]
        vy = X[:, 0:1, :, 4:5]
        phidot = X[:, 0:1, :, 5:6]
        u1 = X[:, 0:1, :, 6:7]
        u2 = X[:, 0:1, :, 7:8]

        f_px = np.concatenate([px, vx*np.cos(phi), vy*np.sin(phi)], axis=-1)
        f_py = np.concatenate([py, vx*np.sin(phi), vy*np.cos(phi)], axis=-1)
        f_phi = np.concatenate([phi, phidot], axis=-1)
        f_vx = np.concatenate([vx, vy*phidot, np.sin(phi), np.cos(phi)], axis=-1)
        f_vy = np.concatenate([vy, vx*phidot, np.sin(phi), np.cos(phi), u1, u2], axis=-1)
        f_phidot = np.concatenate([phidot, u1, u2], axis=-1)

        feature_list = [f_px, f_py, f_phi, f_vx, f_vy, f_phidot]

        # Find maximum feature dimension across all outputs
        max_dim = max([f.shape[-1] for f in feature_list])

        Phi = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))  # (batch, 6, horizon, max_dim)

        for idx, f in enumerate(feature_list):
            Phi[:, [idx], :, :f.shape[-1]] = f
        
        return Phi, [feature[:,0,0,:] for feature in feature_list]
        # return Phi, feature_list

    def BLR_features_grad(self, X):
        # X shape: (batch_size, 2, horizon, 8)
        px = X[:, 0:1, :, 0:1]
        py = X[:, 0:1, :, 1:2]
        phi = X[:, 0:1, :, 2:3]
        vx = X[:, 0:1, :, 3:4]
        vy = X[:, 0:1, :, 4:5]
        phidot = X[:, 0:1, :, 5:6]
        u1 = X[:, 0:1, :, 6:7]
        u2 = X[:, 0:1, :, 7:8]

        ### Now: correct addition rule and product rule

        ## f_px = [px, vx*cos(phi), vy*sin(phi)]
        f_px_grad = np.concatenate([
            np.ones_like(px),                                    # d(px)/d(px)
            vx * (-np.sin(phi)) + np.cos(phi),                   # d(vx*cos(phi))/d(vx,phi)
            vy * np.cos(phi) + np.sin(phi),                      # d(vy*sin(phi))/d(vy,phi)
        ], axis=-1)

        ## f_py = [py, vx*sin(phi), vy*cos(phi)]
        f_py_grad = np.concatenate([
            np.ones_like(py),                                    # d(py)/d(py)
            vx * np.cos(phi) + np.sin(phi),                      # d(vx*sin(phi))/d(vx,phi)
            vy * (-np.sin(phi)) + np.cos(phi),                   # d(vy*cos(phi))/d(vy,phi)
        ], axis=-1)

        ## f_phi = [phi, phidot]
        f_phi_grad = np.concatenate([
            np.ones_like(phi),                                   # d(phi)/d(phi)
            np.ones_like(phidot),                                # d(phidot)/d(phidot)
        ], axis=-1)

        ## f_vx = [vx, vy*phidot, sin(phi), cos(phi)]
        f_vx_grad = np.concatenate([
            np.ones_like(vx),                                    # d(vx)/d(vx)
            vy + phidot,                                                  # d(vy*phidot)/d(vy)                                    # d(vy*phidot)/d(phidot)
            np.cos(phi),                                         # d(sin(phi))/d(phi)
            -np.sin(phi),                                        # d(cos(phi))/d(phi)
        ], axis=-1)

        ## f_vy = [vy, vx*phidot, sin(phi), cos(phi), u1, u2]
        f_vy_grad = np.concatenate([
            np.ones_like(vy),                                    # d(vy)/d(vy)
            vx + phidot,                                               # d(vx*phidot)/d(vx)                                              # d(vx*phidot)/d(phidot)
            np.cos(phi),                                         # d(sin(phi))/d(phi)
            -np.sin(phi),                                        # d(cos(phi))/d(phi)
            np.ones_like(u1),                                    # d(u1)/d(u1)
            np.ones_like(u2),                                    # d(u2)/d(u2)
        ], axis=-1)

        ## f_phidot = [phidot, u1, u2]
        f_phidot_grad = np.concatenate([
            np.ones_like(phidot),                                # d(phidot)/d(phidot)
            np.ones_like(u1),                                    # d(u1)/d(u1)
            np.ones_like(u2),                                    # d(u2)/d(u2)
        ], axis=-1)

        # Collect gradients
        grad_list = [f_px_grad, f_py_grad, f_phi_grad, f_vx_grad, f_vy_grad, f_phidot_grad]

        # Find maximum feature dimension
        max_dim = max(f.shape[-1] for f in grad_list)
        Phi_grad = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))

        for idx, f_grad in enumerate(grad_list):
            Phi_grad[:, [idx], :, :f_grad.shape[-1]] = f_grad

        return Phi_grad

    def ocp_handler(self, func_name, *args, **kwargs):
        """
        Dynamically dispatch OCP function by name.
        
        Args:
            func_name (str): Name of the function to call, e.g., 'cost', 'constraint', etc.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass.
        
        Returns:
            Result of the called function.
        """
        if not hasattr(self, func_name):
            raise ValueError(f"Unknown OCP function requested: {func_name}")

        func = getattr(self, func_name)
        return func(*args, **kwargs)    
    
    def const_expr(self, model_x, num_dyn):
        const_expr = []
        nx = self.params["agent"]["dim"]["nx"]
        v_dim=3
        for i in range(num_dyn):
            xf = np.array(self.params["env"]["terminate_state"])
            xf_dim = xf.shape[0]
            expr = (
                (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf).T
                @ np.array(self.params["optimizer"]["terminal_tightening"]["P"])
                @ (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf)
            )
            const_expr = ca.vertcat(const_expr, expr)
        return None, None
    
    def const_value(self, num_dyn):
        lh = np.empty((0,), dtype=np.float64)
        uh = np.empty((0,), dtype=np.float64)
        delta = self.params["optimizer"]["terminal_tightening"]["delta"]
        lh_e = np.empty((0,), dtype=np.float64)
        uh_e = np.empty((0,), dtype=np.float64)
        return lh, uh, lh_e, uh_e

    def cost_expr(self, model_x, model_u, ns, p, we, optimizer_str):
        pos_dim = 1
        nx = self.params["agent"]["dim"]["nx"]
        nu = self.params["agent"]["dim"]["nu"]
        Qu = np.diag(np.array(self.params[optimizer_str]["Qu"]))
        xg = np.array(self.params["env"]["goal_state"])
        xg_dim = xg.shape[0]
        w = self.params[optimizer_str]["w"]
        Qx = np.diag(np.array(self.params[optimizer_str]["Qx"]))

        
        # # cost
        # if self.params["optimizer"]["cost"] == "mean":
        #     ns = 1
        # else:
        #     ns = self.params["agent"]["num_dyn_samples"]
        expr = 0
        expr_e=0
        v_max = np.array([10,10])
        for i in range(ns):
            translation = model_x[nx * i : nx * (i + 1)][:self.nu] #
            # translation = self.fk(model_x[nx * i : nx * (i + 1)][:self.nu])[:3, 3:]
            expr += (
                (translation - p).T
                @ Qx
                @ (translation - p) + model_x[nx * i : nx * (i + 1)][self.nu:].T@model_x[nx * i : nx * (i + 1)][self.nu:]/1000
                # + (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max).T
                # @ (Qx/50)
                # @ (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max) 
            )
            expr_e += (
                (translation - p).T
                @ Qx
                @ (translation - p)
            )
        cost_expr_ext_cost = expr / ns + model_u.T @ (Qu) @ model_u/100
        cost_expr_ext_cost_e = expr_e / ns
        return cost_expr_ext_cost, cost_expr_ext_cost_e
    

    def cost_expr_variance(self, model_x, model_u, ns, p, p_var, optimizer_str):
        pos_dim = 1
        nx = self.params["agent"]["dim"]["nx"]
        nu = self.params["agent"]["dim"]["nu"]
        Qu = np.diag(np.array(self.params[optimizer_str]["Qu"]))
        xg = np.array(self.params["env"]["goal_state"])
        xg_dim = xg.shape[0]
        w = self.params[optimizer_str]["w"]
        Qx = np.diag(np.array(self.params[optimizer_str]["Qx"]))


        # # === 2. Build CasADi feature functions ===
        # f_px = ca.Function('f_px', [state, control], [self.feature_px(state, control)])
        # f_py = ca.Function('f_py', [state, control], [self.feature_py(state, control)])
        # f_phi = ca.Function('f_phi', [state, control], [self.feature_phi(state, control)])
        # f_vx = ca.Function('f_vx', [state, control], [self.feature_vx(state, control)])
        # f_vy = ca.Function('f_vy', [state, control], [self.feature_vy(state, control)])
        # f_phidot = ca.Function('f_phidot', [state, control], [self.feature_phidot(state, control)])
        f_list = [self.feature_px, self.feature_py, self.feature_phi, self.feature_vx, self.feature_vy, self.feature_phidot]


        # f_list = [f_px, f_py, f_phi, f_vx, f_vy, f_phidot]
        expr = 0
        expr_e=0
        cost = 0
        t_sh = 0
        f_shape = 0
        for feature in f_list:
            t_sh += f_shape
            for i in range(ns):
                f_val = feature(model_x[nx * i : nx * (i + 1)], model_u)
                f_shape = f_val.shape[0]
                expr += -f_val.T @ ca.diag(p_var[t_sh:t_sh+f_shape]) @ f_val
                # expr -= ca.mtimes([f_val.T, ca.diag(p_var[t_sh:t_sh+f_shape]), f_val])
    

        # # expr = 0
        
        # v_max = np.array([10,10])

        # for i in range(ns):
        #     expr += (
        #         (model_x[nx * i : nx * (i + 1)][:xg_dim] - p).T
        #         @ Qx
        #         @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - p))
        #         # + (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max).T
        #         # @ (Qx/50)
        #         # @ (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max) 
        #     )
        #     expr_e += (
        #         (model_x[nx * i : nx * (i + 1)][:xg_dim] - we).T
        #         @ Qx
        #         @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - we)
        #     )
        cost_expr_ext_cost = expr / ns #+ model_u.T @ (Qu) @ model_u
        cost_expr_ext_cost_e = expr_e / ns
        return cost_expr_ext_cost, cost_expr_ext_cost_e

    
    def path_generator(self, st, length=None):
        # Generate values for t from 0 to 2π
        if length is None:
            length = self.params["optimizer"]["H"] + 1
        s = np.linspace(0, 4 * np.pi, 1000)
        t = s[st:st+length] #np.linspace(st + 0, st + 2 * np.pi/100*length, )

        # # Parametric equations for heart
        # x = 16 * np.sin(t)**3
        # y = 10 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

        # Parametric equations for heart
        x = 8 * np.sin(t)**3 / 1.5 + 1
        y = (10 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))/2 - 1 + 1

        traj = np.vstack([x,y]).T
        traj = np.ones_like(traj) * self.params["env"]["goal_state"]
        return traj

    def initialize_plot_handles(self, fig_gp, fig_dyn=None):
        import matplotlib.pyplot as plt
        ax = fig_gp.axes[0]
        ax.set_xlim(-8,8)
        ax.set_ylim(-6, 6)

        ax.grid(which="both", axis="both")
        # ax.minorticks_on()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        y_min = self.params["optimizer"]["x_min"][1]
        y_max = self.params["optimizer"]["x_max"][1]
        x_min = self.params["optimizer"]["x_min"][0]
        x_max = self.params["optimizer"]["x_max"][0]

        # tracking_path = self.path_generator(0, 500)
        # ax.plot(
        #     tracking_path[:, 0],
        #     tracking_path[:, 1],
        #     color="blue",
        #     label="Tracking path",
        # )

        y_min = self.params["optimizer"]["x_min"][1]
        y_max = self.params["optimizer"]["x_max"][1]
        x_min = self.params["optimizer"]["x_min"][0]
        x_max = self.params["optimizer"]["x_max"][0]

        ax.add_line(
            plt.Line2D([x_min, x_max], [y_max, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_max, x_max], [y_min, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_min, x_min], [y_min, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_min, x_max], [y_min, y_min], color="red", linestyle="--")
        )

        if "P" in self.params["optimizer"]["terminal_tightening"]:
            xf = np.array(self.params["env"]["terminate_state"])
            P = np.array(self.params["optimizer"]["terminal_tightening"]["P"])
            delta = self.params["optimizer"]["terminal_tightening"]["delta"]
            L = np.linalg.cholesky(P / delta)
            t = np.linspace(0, 2 * np.pi, 200)
            z = np.vstack([np.cos(t), np.sin(t)])
            ell = np.linalg.inv(L.T) @ z

            # ax.plot(
            #     ell[0, :] + xf[0],
            #     ell[1, :] + xf[1],
            #     color="red",
            #     label="Terminal set",
            # )


        if self.params["env"]["dynamics"] == "bicycle":
            x_max = self.params["optimizer"]["x_max"][0]
            y_ref = self.params["env"]["goal_state"][1]

            ax.add_line(
                plt.Line2D([x_min, x_max], [y_max, y_max], color="red", linestyle="--")
            )
            ax.add_line(
                plt.Line2D([x_min, x_max], [y_min, y_min], color="red", linestyle="--")
            )
            ax.add_line(
                plt.Line2D(
                    [x_min, x_max],
                    [y_ref, y_ref],
                    color="cyan",
                    linestyle=(0, (5, 5)),
                    lw=2,
                )
            )
            # ellipse = Ellipse(xy=(1, 0), width=1.414, height=1,
            #                 edgecolor='r', fc='None', lw=2)
            # ax.add_patch(ellipse)
            if self.params["env"]["ellipses"]:
                for ellipse in self.params["env"]["ellipses"]:
                    x0 = self.params["env"]["ellipses"][ellipse][0]
                    y0 = self.params["env"]["ellipses"][ellipse][1]
                    a_sq = self.params["env"]["ellipses"][ellipse][2]
                    b_sq = self.params["env"]["ellipses"][ellipse][3]
                    f = self.params["env"]["ellipses"][ellipse][4]
                    # u = 1.0  # x-position of the center
                    # v = 0.1  # y-position of the center
                    # f = 0.01
                    a = np.sqrt(a_sq * f)  # radius on the x-axis
                    b = np.sqrt(b_sq * f)  # radius on the y-axis
                    t = np.linspace(0, 2 * np.pi, 100)
                    f2 = 0.5  # plot 2 ellipses, 1 for ego, 1 for other
                    # plt.plot(x0 + a * np.cos(t), y0 + b * np.sin(t))
                    plt.plot(
                        x0 + f2 * a * np.cos(t),
                        y0 + f2 * b * np.sin(t),
                        "black",
                        alpha=0.5,
                    )
                    # plot constarint ellipse
                    plt.plot(x0 + a * np.cos(t), y0 + b * np.sin(t), "gray", alpha=0.5)
                    self.plot_car_stationary(x0, y0, 0, plt)
            # plt.grid(color="lightgray", linestyle="--")
            ax.set_aspect("equal", "box")
            ax.set_xlim(x_min, x_max - 10)
            relax = 0
            ax.set_ylim(y_min - relax, y_max + relax)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.xticks([])
            plt.yticks([])
            plt.xlim([-2.14, 70 + relax])
            plt.tight_layout(pad=0.3)