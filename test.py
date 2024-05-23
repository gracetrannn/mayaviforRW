import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the center, normal, and radius of the circle
# C = np.array([cx, cy, cz])  # replace with the actual coordinates of the center
C = np.array([10, 10, 19])
# N = np.array([nx, ny, nz])  # replace with the actual components of the normal vector
N = np.array([0, 1, 1])
# r = radius  # replace with the actual radius
r = 40

# Normalize the normal vector
N_hat = N / np.linalg.norm(N)

# Find two orthogonal unit vectors in the plane defined by the normal
# This uses the fact that (0,-nz,ny) will be orthogonal to (nx,ny,nz)
U = np.array([0, -N_hat[2], N_hat[1]])
if np.all(U == 0):  # if normal is in the y-direction, choose another U
    U = np.array([-N_hat[2], 0, N_hat[0]])

U_hat = U / np.linalg.norm(U)
V_hat = np.cross(N_hat, U_hat)

# Parametric equation of the circle
theta = np.linspace(0, 2 * np.pi, 100)  # 100 points
circle_points = np.array(
    [C + r * np.cos(t) * U_hat + r * np.sin(t) * V_hat for t in theta])

# Plotting the circle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2])

# Setting equal aspect ratio for all axes
max_val = np.max(np.abs(circle_points))
ax.set_xlim([C[0] - max_val, C[0] + max_val])
ax.set_ylim([C[1] - max_val, C[1] + max_val])
ax.set_zlim([C[2] - max_val, C[2] + max_val])
ax.set_aspect('auto')

plt.show()
