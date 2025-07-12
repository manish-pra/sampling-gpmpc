import numpy as np

class RandomFourierFeatures:
    def __init__(self, D, *, length_scale=1.0, rng=None):
        self.D = D
        self.ls = length_scale
        self.rng = np.random.default_rng(rng)

    def fit(self, X):
        n_features = X.shape[1]
        self.omega = (1.0 / self.ls) * self.rng.normal(size=(self.D, n_features))
        self.phase = self.rng.uniform(0, 2 * np.pi, size=self.D)
        return self

    def transform(self, X):
        proj = X @ self.omega.T + self.phase
        return np.sqrt(2.0 / self.D) * np.cos(proj)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

from urdf2casadi import urdfparser as u2c
import casadi as ca
parser = u2c.URDFparser()
urdf_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/src/environments/urdf/2dof_robot.urdf"
parser.from_file(urdf_path)
# Define symbolic variables
theta = ca.SX.sym("theta", 2)
theta_dot = ca.SX.sym("theta_dot", 2)
tau = ca.SX.sym("tau", 2)

theta_ddot_expr = parser.get_forward_dynamics_aba("base_link", "ee_link")(theta, theta_dot, tau)
forward_dyn = ca.Function("forward_dynamics", [theta, theta_dot, tau], [theta_ddot_expr])

theta_ddot = forward_dyn(theta, theta_dot, tau)  # CasADi symbolic

N_data = 20000
f_batch = forward_dyn.map(N_data)

def robot_dynamics(x0, u, dt):
    x1 = np.zeros_like(x0)
    x1[:,:2] = x0[:,:2] + x0[:,2:4]*dt  
    x1[:,2:4] = x0[:,2:4] + np.array(f_batch(x0[:,:2].T, x0[:,2:4].T, u.T).T)*dt   
    return x1


### Test 

# generate data set
def make_dataset(N=N_data, dt=0.05, rng=None):
    x0 = np.random.uniform(-np.pi/2, np.pi/2, size=(N, 4)) # both position and velocity
    u = np.random.uniform(-1, 1, size=(N, 2)) # actuator on both the motors
    
    x1 = robot_dynamics(x0, u, dt)
    X = np.hstack([x0, u])          # inputs = [state, control]
    y = x1 - x0                     # target = Δx
    return X, y

X, y = make_dataset()
X_train, y_train = X[:16000], y[:16000]
X_test,  y_test  = X[16000:], y[16000:]

# Fit RFF + ridge
rff = RandomFourierFeatures(D=100, length_scale=2.0)
Z_train = rff.fit_transform(X_train)
Z_test = rff.transform(X_test)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1e-8).fit(Z_train, y_train)
# Predict with test data
y_pred = ridge.predict(Z_test)
# get weights and bias
weights = ridge.coef_
bias = ridge.intercept_

# load these features and weights (as true model) and control with acados
# sample multiple weights and biases to get multiple models and control
# Update the model online

# Evaluate the model
from sklearn.metrics import mean_squared_error
print("Test MSE:", mean_squared_error(y_test, y_pred))

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)

plt.scatter(y_test[:, 0], y_pred[:, 0], s=4)
plt.xlabel("True Δx"); plt.ylabel("Pred Δx"); plt.title("X-axis")
plt.subplot(2, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], s=4)
plt.xlabel("True Δy"); plt.ylabel("Pred Δy"); plt.title("Y-axis")
plt.subplot(2, 2, 3)
plt.scatter(y_test[:, 2], y_pred[:, 2], s=4)
plt.xlabel("True Δvx"); plt.ylabel("Pred Δvx"); plt.title("Vx-axis")
plt.subplot(2, 2, 4)
plt.scatter(y_test[:, 3], y_pred[:, 3], s=4)

plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# 5. Quick visual check
# ------------------------------------------------------------
# for j, qj in zip(joint_indices, q):
#     p.setJointMotorControl2(
#         robot_id, j, p.POSITION_CONTROL, targetPosition=qj, force=500
#     )
#     p.stepSimulation()
# #     p.setJointMotorControl2(robot_id, j, controlMode=p.POSITION_CONTROL, targetPosition=qj, force=5.0)