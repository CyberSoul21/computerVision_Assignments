# ==============================================
#   LAB SESSION 2 – Point Triangulation
#   Author: Diego Méndez (guided version)
#   Topic: Compute P1, P2 and triangulate 3D points
# ==============================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define base path for your Lab files
# ---------------------------------------------------------
base_path = "D:/Dev_Space/python/Lab2/"

# ---------------------------------------------------------
# 2. Load extrinsic and intrinsic parameters
# ---------------------------------------------------------
T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 matrix: camera1 -> world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 matrix: camera2 -> world
K_c     = np.loadtxt(base_path + "K_c.txt")     # 3x3 intrinsic matrix

# ---------------------------------------------------------
# 3. Invert transformations (we need world -> camera)
# ---------------------------------------------------------
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)

# ---------------------------------------------------------
# 4. Build projection matrices P = K [R | t]
# ---------------------------------------------------------
R1 = T_c1_w[0:3, 0:3]
t1 = T_c1_w[0:3, 3].reshape(3, 1)
R2 = T_c2_w[0:3, 0:3]
t2 = T_c2_w[0:3, 3].reshape(3, 1)

P1 = K_c @ np.hstack((R1, t1))
P2 = K_c @ np.hstack((R2, t2))

print("=== Projection Matrix P1 ===")
print(P1)
print("\n=== Projection Matrix P2 ===")
print(P2)

# ---------------------------------------------------------
# 5. Load matched 2D points (each file has 2 rows: u and v)
# ---------------------------------------------------------
x1 = np.loadtxt(base_path + "x1Data.txt")
x2 = np.loadtxt(base_path + "x2Data.txt")

print("\nLoaded", x1.shape[1], "matches.")
print("x1 shape:", x1.shape, "| x2 shape:", x2.shape)

# ---------------------------------------------------------
# 6. Define the triangulation function (A X = 0 -> SVD)
# ---------------------------------------------------------
def triangulate_point(u1, v1, u2, v2, P1, P2):
    """
    Triangulate one 3D point from two image correspondences.
    Solves A X = 0 using SVD, taking the last column of V.
    """
    A = np.array([
        u1 * P1[2, :] - P1[0, :],
        v1 * P1[2, :] - P1[1, :],
        u2 * P2[2, :] - P2[0, :],
        v2 * P2[2, :] - P2[1, :]
    ])

    # --- Solve A X = 0 using SVD ---
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]              # last row of Vt = last column of V
    X /= X[3]               # normalize homogeneous coord


    return X[:3]             # return (X, Y, Z)

# ---------------------------------------------------------
# 7. Triangulate all points
# ---------------------------------------------------------
points_3D = []
for i in range(x1.shape[1]):   # recorrer columnas (cada punto)
    u1, v1 = x1[0, i], x1[1, i]
    u2, v2 = x2[0, i], x2[1, i]
    X = triangulate_point(u1, v1, u2, v2, P1, P2)
    points_3D.append(X)

points_3D = np.array(points_3D)
print("\nTriangulated", points_3D.shape[0], "points.")

# ---------------------------------------------------------
# 8. Plot the resulting 3D points
# ---------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2],
           c='r', marker='.', s=5, label='Triangulated Points')

# plot camera centers
C1_w = T_w_c1[0:3, 3]
C2_w = T_w_c2[0:3, 3]
ax.scatter(C1_w[0], C1_w[1], C1_w[2], color='blue', s=40, label='Camera 1')
ax.scatter(C2_w[0], C2_w[1], C2_w[2], color='green', s=40, label='Camera 2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Triangulated 3D Points and Camera Poses')

plt.show()
