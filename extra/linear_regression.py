import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
def generate_data(n_samples=100):
    X = np.linspace(-3, 3, n_samples)
    y = 0.5 * X**3 - X**2 + 2 * X + np.random.randn(n_samples) * 3  # Nonlinear relation + noise
    return X[:, np.newaxis], y

# Feature expansion (finite nonlinear features)
def expand_features(X):
    """
    X: shape (n_samples, 1)
    Returns expanded features: [1, x, x^2, x^3]
    """
    return np.hstack([
        np.ones((X.shape[0], 1)),  # Bias term
        X,
        X**2,
        X**3,
        np.sin(X),
        np.cos(X),
        np.exp(X)
    ])

# Linear Regression using closed-form solution
def train_linear_regression(X, y):
    """
    Solves (X^T X)^-1 X^T y
    """
    XTX_inv = np.linalg.inv(X.T @ X)
    w = XTX_inv @ X.T @ y
    return w

def train_ridge_regression(X, y, lambda_reg=1e-3):
    """
    Ridge Regression: (X^T X + lambda * I)^-1 X^T y
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    XTX_plus_lambdaI = X.T @ X + lambda_reg * I
    w = np.linalg.inv(XTX_plus_lambdaI) @ X.T @ y
    return w

def train_bayesian_linear_regression(X, y, lambda_reg=1e-3, noise_var=1.0):
    n_features = X.shape[1]
    I = np.eye(n_features)
    A = X.T @ X + lambda_reg * I
    A_inv = np.linalg.inv(A)
    
    mu = A_inv @ X.T @ y
    Sigma = noise_var * A_inv
    return mu, Sigma

def predict_bayesian(X_test, mu, Sigma, noise_var=1.0):
    mean_pred = X_test @ mu
    var_pred = np.sum(X_test @ Sigma * X_test, axis=1) + noise_var
    return mean_pred, var_pred


# Prediction function
def predict(X, w):
    return X @ w

# Main workflow
X_raw, y = generate_data()
X_expanded = expand_features(X_raw)

mu, Sigma = train_bayesian_linear_regression(X_expanded, y, lambda_reg=0.01, noise_var=9.0)  # assuming std ~ 3
y_mean, y_var = predict_bayesian(X_expanded, mu, Sigma, noise_var=9.0)
y_std = np.sqrt(y_var)

plt.scatter(X_raw, y, label="Data", alpha=0.6)
plt.plot(X_raw, y_mean, color='red', label="Mean prediction")
plt.fill_between(X_raw.flatten(), 
                 y_mean - 2 * y_std, 
                 y_mean + 2 * y_std, 
                 color='red', alpha=0.2, label="Uncertainty (2 std)")
plt.legend()
plt.title("Bayesian Linear Regression: Mean + Uncertainty")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# # w = train_linear_regression(X_expanded, y)
# w = train_ridge_regression(X_expanded, y, lambda_reg=0.0001)
# y_pred = predict(X_expanded, w)

# # Visualization
# plt.scatter(X_raw, y, label="Data", alpha=0.6)
# plt.plot(X_raw, y_pred, color='red', label="Model prediction")
# plt.legend()
# plt.title("Linear Regression with Nonlinear Features")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()

# # Print model parameters
# print("Learned weights:", w)
