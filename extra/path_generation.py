import numpy as np
import matplotlib.pyplot as plt

# Generate values for t from 0 to 2π
# t = np.linspace(0, 2 * np.pi, 100)
st = 0
length = 100
t = np.linspace(st + 0, st + 2 * np.pi/100*length, length)

# Parametric equations for heart
x = 8 * np.sin(t)**3 / 1.5
y = (10 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))/2 - 1

# Plot
plt.plot(x, y, '.',color='red')
plt.title('Heart Shape')
plt.axis('equal')
plt.grid(True)
plt.show()

# Coordinates
coordinates = list(zip(x, y))