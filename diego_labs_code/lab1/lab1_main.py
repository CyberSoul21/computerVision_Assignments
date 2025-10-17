# lab1_main.py
"""
Minimal runner that matches your folder layout:
- K.txt
- R_w_c1.txt, t_w_c1.txt
- R_w_c2.txt, t_w_c2.txt
- (optional) x2DGTL***.txt for 2D points used in SVD line fitting
- (optional) points3D.txt with rows "X Y Z" (A,B,C,D,E,...)
Usage:
    python lab1_main.py
Edit the "DEMO_*" blocks to plug in your data.
"""

import os
import numpy as np
from lab1_utils import load_cameras, project, line_through, intersect, vanishing_point, fit_line_svd, plane_from_points, point_plane_distance

# ---------- Load cameras ----------
try:
    cams = load_cameras("D:/Dev_Space/python/Practice1/")
except Exception as e:
    print("⚠️  No pude cargar las cámaras. Asegúrate de tener K.txt, R_w_c1.txt, t_w_c1.txt, R_w_c2.txt, t_w_c2.txt aquí.")
    print("Error:", e)
    cams = None

# ---------- DEMO: 3D points (A,B,C,D,E,...) ----------
# If you already have them in a file (rows: X Y Z), just uncomment next lines:
# pts3d = np.loadtxt("points3D.txt")   # shape (N,3)
# names = [f"P{i}" for i in range(pts3d.shape[0])]

# For now, we put a small placeholder square on Z=0.82 (you should replace with your real points)
pts3d = np.array([[3.44, 0.80, 0.82],
                  [4.20, 0.80,0.82 ],
                  [4.20, 0.60, 0.82],
                  [3.55, 0.60, 0.82],
                  [-0.01, 2.6, 1.21]], dtype=float)
names = ["A","B","C","D","E"]

if cams:
    # Project onto image 1 & 2
    x1 = project(cams.P1, pts3d)
    x2 = project(cams.P2, pts3d)
    print("\n== Proyecciones (imagen 1) ==")
    for n, p in zip(names, x1):
        print(f"{n}: ({p[0]:.2f}, {p[1]:.2f})")
    print("\n== Proyecciones (imagen 2) ==")
    for n, p in zip(names, x2):
        print(f"{n}: ({p[0]:.2f}, {p[1]:.2f})")

    # ---------- Vanishing point of direction (B-A) ----------
    v = pts3d[1] - pts3d[0]
    vp1 = vanishing_point(cams.P1, v)
    vp2 = vanishing_point(cams.P2, v)
    print("\nPunto de fuga de AB:")
    print(" img1:", vp1)
    print(" img2:", vp2)

# ---------- DEMO: SVD line fitting with given 2D pixels ----------
# If you have a file with 2D points per row: x y
if os.path.exists("x2DLineFittingSVD.txt"):
    pts2d = np.loadtxt("x2DLineFittingSVD.txt")
    l = fit_line_svd(pts2d)
    print("\nRecta estimada (ax+by+c=0) con SVD:", l)
    # If you have two extreme points in another file, you can compare:
    if os.path.exists("x2DGTLineFittingSVD.txt"):
        p = np.loadtxt("x2DGTLineFittingSVD.txt")  # 2x2
        l_gt = np.cross(np.r_[p[0],1.0], np.r_[p[1],1.0])
        l_gt = l_gt / np.linalg.norm(l_gt[:2])
        print("Recta GT (normalizada):", l_gt)

# ---------- DEMO: plane & distances ----------
pi = plane_from_points(pts3d[:4])   # plane through A,B,C,D
dE = point_plane_distance(pts3d[4], pi)
print("\nPlano ABCD (a,b,c,d):", pi)
print("Distancia de E al plano ABCD:", f"{dE:.3f}", "unidades")
