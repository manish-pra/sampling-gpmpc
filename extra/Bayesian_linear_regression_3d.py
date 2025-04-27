import numpy as np
import matplotlib.pyplot as plt

# 1. Generate 2D data
def generate_data_2d(n_samples=100):
    X = np.random.uniform(-3, 3, size=(n_samples, 2))  # 2D input
    y = 2 * X[:, 0]**2 - 3 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + np.random.randn(n_samples) * 3
    return X, y


def generate_data_2d_multimodal(n_samples=100):
    """
    Generate 2D data clustered around multiple centers.
    """
    centers = np.array([
        [-2, -2],
        [2, 2],
        [-2, 2],
        [2, -2]
    ])
    
    X = []
    y = []
    
    for _ in range(n_samples):
        center = centers[np.random.choice(len(centers))]
        point = center + np.random.randn(2) * 0.5  # small random noise around center
        X.append(point)
        
        # Define a true function (nonlinear)
        y_point = 2 * point[0]**2 - 3 * point[1] + 0.5 * point[0] * point[1] + np.random.randn() * 3
        y.append(y_point)
    
    X = np.array(X)
    y = np.array(y)
    return X, y

# 2. Expand 2D features nonlinearly
def expand_features_2d(X):
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    return np.hstack([
        np.ones((X.shape[0], 1)),
        x1,
        x2,
        x1**2,
        x2**2,
        x1 * x2
    ])

# 3. Train Bayesian Linear Regression
def train_bayesian_linear_regression(X, y, lambda_reg=1e-3, noise_var=1.0):
    n_features = X.shape[1]
    I = np.eye(n_features)
    A = X.T @ X + lambda_reg * I
    A_inv = np.linalg.inv(A)
    mu = A_inv @ X.T @ y
    Sigma = noise_var * A_inv
    return mu, Sigma

# 4. Predict with uncertainty
def predict_bayesian(X_test, mu, Sigma, noise_var=1.0):
    mean_pred = X_test @ mu
    var_pred = np.sum(X_test @ Sigma * X_test, axis=1) + noise_var
    return mean_pred, var_pred

# 5. Main script
if __name__ == "__main__":
    # Generate and expand data
    # X_raw, y = generate_data_2d()
    X_raw, y = generate_data_2d_multimodal()
    X_expanded = expand_features_2d(X_raw)

    # Train model
    mu, Sigma = train_bayesian_linear_regression(X_expanded, y, lambda_reg=0.01, noise_var=9.0)

    # Predict
    y_mean, y_var = predict_bayesian(X_expanded, mu, Sigma, noise_var=9.0)
    y_std = np.sqrt(y_var)

    # Create a grid over the 2D space
    x1_grid = np.linspace(-3, 3, 50)
    x2_grid = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T  # shape (2500, 2)

    # Expand features
    X_grid_expanded = expand_features_2d(X_grid)

    # Predict on the grid
    y_mean_grid, y_var_grid = predict_bayesian(X_grid_expanded, mu, Sigma, noise_var=9.0)
    y_std_grid = np.sqrt(y_var_grid)
    y_std_grid = y_std_grid.reshape(X1.shape)

    # Plot uncertainty as a contour plot
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_std_grid, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Predictive Std (Uncertainty)')
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c='white', edgecolors='k', label='Training data')
    plt.title('Predictive Uncertainty (Standard Deviation)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
