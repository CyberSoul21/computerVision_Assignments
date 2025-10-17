# ==============================================
#   LAB SESSION 2 – COMPLETE SCRIPT
#   Feature Detection and Matching / Epipolar Geometry
#   Diego Méndez – MRGCV / Computer Vision Lab
# ==============================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. Define base path for your Lab files
# ---------------------------------------------------------
base_path = "D:/Dev_Space/python/Lab2/"

# ---------------------------------------------------------
# 2. Load extrinsic and intrinsic parameters
# ---------------------------------------------------------
T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 camera1 → world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 camera2 → world
K_c     = np.loadtxt(base_path + "K_c.txt")     # 3x3 intrinsics

# ---------------------------------------------------------
# 3. Invert transformations (world → camera)
# ---------------------------------------------------------
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)

# ---------------------------------------------------------
# 4. Build projection matrices P = K [R | t]
# ---------------------------------------------------------
R1, t1 = T_c1_w[:3, :3], T_c1_w[:3, 3].reshape(3, 1)
R2, t2 = T_c2_w[:3, :3], T_c2_w[:3, 3].reshape(3, 1)

P1 = K_c @ np.hstack((R1, t1))
P2 = K_c @ np.hstack((R2, t2))

np.set_printoptions(precision=4, suppress=True)
print("\n========== MATRICES DE PROYECCIÓN ==========")
print("P1 = K[R1|t1]:\n", P1)
print("\nP2 = K[R2|t2]:\n", P2)

# ---------------------------------------------------------
# 5. Load matched 2D points (each file has 2 rows: u and v)
# ---------------------------------------------------------
x1 = np.loadtxt(base_path + "x1Data.txt")
x2 = np.loadtxt(base_path + "x2Data.txt")
print("\nLoaded", x1.shape[1], "matches.")

# ---------------------------------------------------------
# 6. Triangulation function (A X = 0 -> SVD)
# ---------------------------------------------------------
def triangulate_point(u1, v1, u2, v2, P1, P2):
    A = np.array([
        u1 * P1[2, :] - P1[0, :],
        v1 * P1[2, :] - P1[1, :],
        u2 * P2[2, :] - P2[0, :],
        v2 * P2[2, :] - P2[1, :]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3]

# ---------------------------------------------------------
# 7. Triangulate all points
# ---------------------------------------------------------
points_3D = np.array([
    triangulate_point(x1[0, i], x1[1, i], x2[0, i], x2[1, i], P1, P2)
    for i in range(x1.shape[1])
])
print("\nTriangulated", points_3D.shape[0], "points.")

# ---------------------------------------------------------
# 8. Compute Fundamental Matrix from poses
# ---------------------------------------------------------
def fundamental_from_poses(T_c1_w, T_c2_w, K):
    T_21 = T_c2_w @ np.linalg.inv(T_c1_w)
    R, t = T_21[:3, :3], T_21[:3, 3]
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ t_x @ R @ np.linalg.inv(K)
    return F / F[2, 2]

F21 = fundamental_from_poses(T_c1_w, T_c2_w, K_c)
print("\n========== FUNDAMENTAL MATRIX F21 ==========")
print(F21)

# ---------------------------------------------------------
# 9. Verify Epipolar Constraint
# ---------------------------------------------------------
total_error = 0
for i in range(x1.shape[1]):
    x1_h = np.array([x1[0, i], x1[1, i], 1])
    x2_h = np.array([x2[0, i], x2[1, i], 1])
    total_error += abs(x2_h.T @ F21 @ x1_h)
mean_error = total_error / x1.shape[1]
print(f"\nMean epipolar constraint error: {mean_error:.6e}")

# ---------------------------------------------------------
# 10. Visualize Epipolar Lines (l0 and l1)
# ---------------------------------------------------------
img1 = cv2.imread(base_path + "image1.png")
img2 = cv2.imread(base_path + "image2.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def draw_epipolar_line(img, l, color):
    a, b, c = l
    h, w = img.shape[:2]
    x0, y0 = 0, int(-c / b)
    x1, y1 = w, int(-(c + a*w) / b)
    img_line = img.copy()
    cv2.line(img_line, (x0, y0), (x1, y1), color, 1)
    return img_line

# Draw a few example epipolar lines
num_points = 8
indices = random.sample(range(x1.shape[1]), num_points)
img1_draw, img2_draw = img1.copy(), img2.copy()

for i in indices:
    x1_h = np.array([x1[0, i], x1[1, i], 1])
    x2_h = np.array([x2[0, i], x2[1, i], 1])
    l0 = F21 @ x1_h     # line in image 2
    l1 = F21.T @ x2_h   # line in image 1
    color = tuple(np.random.randint(0, 255, 3).tolist())
    img1_draw = draw_epipolar_line(img1_draw, l1, color)
    img2_draw = draw_epipolar_line(img2_draw, l0, color)
    cv2.circle(img1_draw, (int(x1[0, i]), int(x1[1, i])), 5, color, -1)
    cv2.circle(img2_draw, (int(x2[0, i]), int(x2[1, i])), 5, color, -1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(img1_draw)
ax1.set_title("Image 1 with Epipolar Lines (l1 = F^T x2)")
ax2.imshow(img2_draw)
ax2.set_title("Image 2 with Epipolar Lines (l0 = F x1)")
plt.show()

# ---------------------------------------------------------
# 11. INTERACTIVE MODE: Click on Image 1 → Line in Image 2
# ---------------------------------------------------------
def onclick_image1(event):
    """Callback when user clicks on image 1."""
    if event.inaxes != ax1_click:
        return

    u1, v1 = event.xdata, event.ydata
    print(f"\nClicked point in Image 1: ({u1:.2f}, {v1:.2f})")

    x1_h = np.array([u1, v1, 1.0])
    l0 = F21 @ x1_h
    print("Epipolar line coefficients in Image 2:", l0)

    img1_click = img1.copy()
    cv2.circle(img1_click, (int(u1), int(v1)), 6, (255, 0, 0), -1)
    img2_line = draw_epipolar_line(img2, l0, (255, 0, 0))

    fig2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    ax1b.imshow(img1_click)
    ax1b.set_title(f"Clicked Point in Image 1 ({u1:.1f}, {v1:.1f})")
    ax2b.imshow(img2_line)
    ax2b.set_title("Epipolar Line in Image 2 (l0 = F x1)")
    plt.show()

# Set up figure for interaction
fig, (ax1_click, ax2_click) = plt.subplots(1, 2, figsize=(14, 6))
ax1_click.imshow(img1)
ax1_click.set_title("Click on a point in Image 1")
ax2_click.imshow(img2)
ax2_click.set_title("Image 2 (Epipolar line will appear here)")
cid = fig.canvas.mpl_connect('button_press_event', onclick_image1)
plt.show()
