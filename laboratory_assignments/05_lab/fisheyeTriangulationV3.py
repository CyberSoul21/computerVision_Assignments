#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 5
#
# Title: Laboratory 5: 3D reconstruction from 3D calibrated stereo using fish-eyes.
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################


# ############################################################################
# TRIANGULATION OF TWO RAYS (PLANE-BASED / LINE-BASED)
# ############################################################################

import numpy as np
from kannalaBrandt import (projectKannalaBrandt, 
                            unprojectKannalaBrandt)

basePath = "labSession5/"

def triangulate_two_rays(C1, d1, C2, d2, eps=1e-9):
    """
    Triangulate a 3D point from two camera rays in WORLD coordinates.

    Ray 1: X = C1 + lambda1 * d1
    Ray 2: X = C2 + lambda2 * d2

    We find the closest points on each ray and return the midpoint.

    Inputs:
      C1, C2 : (3,) camera centers in WORLD coordinates
      d1, d2 : (3,) direction vectors in WORLD coordinates (not necessarily unit,
               but it is better if they are)

    Output:
      X      : (3,) triangulated 3D point in WORLD frame
      P1, P2 : (3,) closest points on each ray (for debugging / quality checks)
    """
    C1 = np.asarray(C1, dtype=float).reshape(3)
    C2 = np.asarray(C2, dtype=float).reshape(3)
    d1 = np.asarray(d1, dtype=float).reshape(3)
    d2 = np.asarray(d2, dtype=float).reshape(3)

    # Normalize directions to avoid scale issues
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < eps or n2 < eps:
        raise ValueError("Direction vectors must be non-zero")
    d1 = d1 / n1
    d2 = d2 / n2

    w0 = C1 - C2
    a = np.dot(d1, d1)       # ~1
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)       # ~1
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b

    if abs(denom) < eps:
        # Rays (almost) parallel: just project C2 onto ray1 as fallback
        lambda1 = -d / (a + eps)
        lambda2 = 0.0
    else:
        lambda1 = (b * e - c * d) / denom
        lambda2 = (a * e - b * d) / denom

    P1 = C1 + lambda1 * d1
    P2 = C2 + lambda2 * d2
    X = 0.5 * (P1 + P2)
    return X, P1, P2



def triangulate_poseA_kb(
    x1, x2,
    K1, D1, K2, D2,
    T_wc1, T_wc2,
    normalize_rays=True
):
    """
    Triangulate 3D points for pose A using a calibrated fisheye stereo pair.

    Inputs:
      x1, x2 : (N,2) arrays with pixel coordinates in camera 1 and 2
      K1, D1 : intrinsics and distortion for camera 1
      K2, D2 : intrinsics and distortion for camera 2
      T_wc1  : (4,4) homogeneous transform from camera 1 frame to WORLD (rig) frame
      T_wc2  : (4,4) homogeneous transform from camera 2 frame to WORLD (rig) frame

    Output:
      Xw     : (N,3) array of 3D points in WORLD frame
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    assert x1.shape == x2.shape
    N = x1.shape[0]

    # 1) Rays in each camera frame via Kannala-Brandt unprojection
    rays1_cam = unprojectKannalaBrandt(x1, K1, D1, normalize=normalize_rays)  # (N,3)
    rays2_cam = unprojectKannalaBrandt(x2, K2, D2, normalize=normalize_rays)  # (N,3)

    # 2) Extract R_wc and C from T_wc (X_w = R_wc * X_c + t_wc)
    T_wc1 = np.asarray(T_wc1, dtype=float).reshape(4, 4)
    T_wc2 = np.asarray(T_wc2, dtype=float).reshape(4, 4)

    Rwc1 = T_wc1[:3, :3]
    twc1 = T_wc1[:3, 3]      # camera center 1 in world frame
    Rwc2 = T_wc2[:3, :3]
    twc2 = T_wc2[:3, 3]      # camera center 2 in world frame

    C1 = twc1
    C2 = twc2

    # 3) Rays in WORLD frame
    d1_world = (Rwc1 @ rays1_cam.T).T   # (N,3)
    d2_world = (Rwc2 @ rays2_cam.T).T   # (N,3)

    # 4) Triangulate each pair of rays
    Xw = np.zeros((N, 3), dtype=float)
    for i in range(N):
        X, _, _ = triangulate_two_rays(C1, d1_world[i], C2, d2_world[i])
        Xw[i, :] = X

    return Xw


def main():
    # ------------------------------------------------------------
    # 1) Load image correspondences for pose A
    #     x1.txt, x2.txt are 3 x N (u; v; 1)
    # ------------------------------------------------------------
    X1 = np.loadtxt(basePath + "x1.txt")  # shape (3, N)
    X2 = np.loadtxt(basePath + "x2.txt")  # shape (3, N)

    # convert to (N,2) pixel arrays
    x1 = np.vstack((X1[0, :], X1[1, :])).T   # (N,2)
    x2 = np.vstack((X2[0, :], X2[1, :])).T   # (N,2)

    print("Loaded", x1.shape[0], "matches for pose A.")

    # ------------------------------------------------------------
    # 2) Load intrinsics and distortion (adjust filenames if needed)
    # ------------------------------------------------------------
    K1 = np.loadtxt(basePath + "K_1.txt")
    K2 = np.loadtxt(basePath + "K_2.txt")
    D1_all = np.loadtxt(basePath + "D1_k_array.txt")
    D2_all = np.loadtxt(basePath + "D2_k_array.txt")
    D1 = D1_all[:4]
    D2 = D2_all[:4]

    # ------------------------------------------------------------
    # 3) Load T_wc1, T_wc2 (4x4 homogeneous matrices)
    # ------------------------------------------------------------
    T_wc1 = np.loadtxt(basePath + "T_wc1.txt")
    T_wc2 = np.loadtxt(basePath + "T_wc2.txt")

    # ------------------------------------------------------------
    # 4) Triangulate
    # ------------------------------------------------------------
    Xw = triangulate_poseA_kb(
        x1, x2,
        K1, D1, K2, D2,
        T_wc1, T_wc2
    )

    print("First 5 3D points (WORLD frame, pose A):")
    print(Xw[:5])

    # ------------------------------------------------------------
    # 5) Save 3D points to file for later visualization / BA
    # ------------------------------------------------------------
    np.savetxt("points3D_poseA.txt", Xw)
    print("Saved 3D points to points3D_poseA.txt")

if __name__ == "__main__":
    main()

    # ------------------------------------------------------------
    # 1) Load triangulated 3D points (WORLD frame)
    # ------------------------------------------------------------
    Xw = np.loadtxt("points3D_poseA.txt")   # (N,3)
    N = Xw.shape[0]
    print("Loaded", N, "3D points from points3D_poseA.txt") 
    # ------------------------------------------------------------
    # 2) Load original correspondences x1, x2 (pixels)
    #     x1.txt, x2.txt are 3 x N with [u; v; 1]
    # ------------------------------------------------------------
    X1 = np.loadtxt(basePath + "x1.txt")   # (3,N)
    X2 = np.loadtxt(basePath + "x2.txt")   # (3,N)
    x1 = np.vstack((X1[0, :], X1[1, :])).T   # (N,2)
    x2 = np.vstack((X2[0, :], X2[1, :])).T   # (N,2)    
    # ------------------------------------------------------------
    # 3) Load intrinsics + distortion
    # ------------------------------------------------------------
    K1 = np.loadtxt(basePath + "K_1.txt")
    K2 = np.loadtxt(basePath + "K_2.txt")
    D1_all = np.loadtxt(basePath + "D1_k_array.txt")
    D2_all = np.loadtxt(basePath + "D2_k_array.txt")
    D1 = D1_all[:4]
    D2 = D2_all[:4] 
    # ------------------------------------------------------------
    # 4) Load camera poses T_wc1, T_wc2 and invert to get T_cw
    # ------------------------------------------------------------
    T_wc1 = np.loadtxt(basePath + "T_wc1.txt")
    T_wc2 = np.loadtxt(basePath + "T_wc2.txt") 
    # Inverse: X_c = R_cw * X_w + t_cw
    T_cw1 = np.linalg.inv(T_wc1)
    T_cw2 = np.linalg.inv(T_wc2)    
    R_cw1 = T_cw1[:3, :3]
    t_cw1 = T_cw1[:3, 3]
    R_cw2 = T_cw2[:3, :3]
    t_cw2 = T_cw2[:3, 3]    
    # ------------------------------------------------------------
    # 5) Transform 3D points into each camera frame
    # ------------------------------------------------------------
    # Xc = R_cw * Xw + t_cw   (apply to each point)
    Xc1 = (R_cw1 @ Xw.T + t_cw1.reshape(3,1)).T   # (N,3)
    Xc2 = (R_cw2 @ Xw.T + t_cw2.reshape(3,1)).T   # (N,3)   
    depth1 = Xc1[:, 2]
    depth2 = Xc2[:, 2]  
    print("\n=== Depth statistics (camera 1 frame) ===")
    print("min Z:", depth1.min(), " max Z:", depth1.max())  
    print("\n=== Depth statistics (camera 2 frame) ===")
    print("min Z:", depth2.min(), " max Z:", depth2.max())  
    # Ideally, min Z should be > 0 in both cameras
    num_behind1 = np.sum(depth1 <= 0)
    num_behind2 = np.sum(depth2 <= 0)
    print("\nPoints behind cam1 (Z<=0):", int(num_behind1))
    print("Points behind cam2 (Z<=0):", int(num_behind2))   
    # ------------------------------------------------------------
    # 6) Reproject points into each image with KB model
    # ------------------------------------------------------------
    u1_pred = projectKannalaBrandt(Xc1, K1, D1)   # (N,2)
    u2_pred = projectKannalaBrandt(Xc2, K2, D2)   # (N,2) 
    err1 = np.linalg.norm(u1_pred - x1, axis=1)
    err2 = np.linalg.norm(u2_pred - x2, axis=1) 
    print("\n=== Reprojection error cam1 (pixels) ===")
    print("mean:", err1.mean(), "  max:", err1.max())   
    print("\n=== Reprojection error cam2 (pixels) ===")
    print("mean:", err2.mean(), "  max:", err2.max())   
    # Optional: print per-point for debugging
    # for i in range(N):
    #     print(f"Pt {i:2d}: err1={err1[i]:.4f}, err2={err2[i]:.4f}")    