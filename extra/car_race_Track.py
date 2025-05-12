import numpy as np
import matplotlib.pyplot as plt

def ellipse_points(P, c, num_points=300):
    """Generate points on the ellipse defined by (x - c)^T P (x - c) = 1."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)])  # Shape: (2, N)

    # Transform unit circle to match ellipse shape
    L = np.linalg.cholesky(np.linalg.inv(P))  # such that x = L @ u gives the ellipse
    ellipse = (L @ unit_circle).T + c
    return ellipse

def plot_ellipse_ring(P_outer, P_inner, c):
    """
    Plot a racetrack ring using two different ellipse matrices (inner and outer).
    
    Parameters:
    - P_outer: 2x2 positive-definite matrix for outer ellipse
    - P_inner: 2x2 positive-definite matrix for inner ellipse
    - c: 2D center
    """
    outer = ellipse_points(P_outer, c)
    inner = ellipse_points(P_inner, c)

    plt.figure(figsize=(8, 6))
    plt.plot(outer[:, 0], outer[:, 1], 'k-', label='Outer Boundary')
    plt.plot(inner[:, 0], inner[:, 1], 'k-', label='Inner Boundary')
    plt.fill(outer[:, 0], outer[:, 1], color='gray', alpha=0.3)
    plt.fill(inner[:, 0], inner[:, 1], color='white', alpha=1.0)
    plt.axis('equal')
    plt.title("Racetrack with Two Different Ellipse Matrices")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
# Outer ellipse: wider
a_out, b_out = 120, 40
P_outer = np.diag([1/a_out**2, 1/b_out**2])

# Inner ellipse: narrower, maybe tilted
f = 0.9
a_in, b_in = 65/f, 15/f
# angle = np.deg2rad(20)
# R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
# D = np.diag([1/a_in**2, 1/b_in**2])
# P_inner = R @ D @ R.T  # rotated ellipse
P_inner = np.diag([1/a_in**2, 1/b_in**2])

c = np.array([0, 0])

plot_ellipse_ring(P_outer, P_inner, c)