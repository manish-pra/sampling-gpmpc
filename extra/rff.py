# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. RFF transformer (same as before, but handles any dim)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Toy robot dynamics  x_{t+1} = x_t + g(x_t,u_t)·Δt
# ------------------------------------------------------------
def robot_dynamics(x, u, dt=0.05):
    """
    x : (n,2) position
    u : (n,2) control force  (bounded in [-1,1])
    """
    mass = 1.0
    # nonlinear drag: force ∝ |v| v
    drag_coef = 0.4
    v = u / mass
    drag = drag_coef * np.linalg.norm(v, axis=1, keepdims=True) * v
    accel = (u - drag) / mass
    return x + accel * dt

# ------------------------------------------------------------
# 3. Generate data
# ------------------------------------------------------------
def make_dataset(N=20000, dt=0.05, rng=None):
    rng = np.random.default_rng(rng)
    x0 = rng.uniform(-2, 2, size=(N, 2))
    u = rng.uniform(-1, 1, size=(N, 2))
    x1 = robot_dynamics(x0, u, dt)
    X = np.hstack([x0, u])          # inputs = [state, control]
    y = x1 - x0                     # target = Δx
    return X, y

X, y = make_dataset()
X_train, y_train = X[:16000], y[:16000]
X_test,  y_test  = X[16000:], y[16000:]

# ------------------------------------------------------------
# 4. Fit RFF + ridge
# ------------------------------------------------------------
rff = RandomFourierFeatures(D=10, length_scale=1.0, rng=42)
Z_train = rff.fit_transform(X_train)
Z_test  = rff.transform(X_test)

ridge = Ridge(alpha=1e-3).fit(Z_train, y_train)
y_pred = ridge.predict(Z_test)

print("Test MSE:", mean_squared_error(y_test, y_pred))

# ------------------------------------------------------------
# 5. Quick visual check
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(y_test[:,0], y_pred[:,0], s=4)
ax[0].set_xlabel("True Δx"); ax[0].set_ylabel("Pred Δx"); ax[0].set_title("X-axis")
ax[1].scatter(y_test[:,1], y_pred[:,1], s=4)
ax[1].set_xlabel("True Δy"); ax[1].set_ylabel("Pred Δy"); ax[1].set_title("Y-axis")
for a in ax: a.plot([-0.2,0.2],[-0.2,0.2],'k--')
plt.tight_layout(); plt.show()