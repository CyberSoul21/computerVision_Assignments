#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 5
#
# Title: Methods for laboratory 5.
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################
"""
Kannala-Brandt Camera Model for Fisheye Lenses

This module implements the projection and unprojection functions
for fisheye cameras using the Kannala-Brandt distortion model.

References:
- Kannala, J., & Brandt, S. (2006). "A generic camera model and 
  calibration method for conventional, wide-angle, and fish-eye lenses."
"""

import numpy as np
basePath = "labSession5/"

# ############################################################################
# KANNALA-BRANDT FISHEYE PROJECTION / UNPROJECTION
# ############################################################################

def projectKannalaBrandt(Pc, K, D):
    """
    Project 3D camera-frame points using the Kannala-Brandt fisheye model
    (OpenCV fisheye-style).
    
    Inputs:
      Pc : (3,) or (N,3) array of 3D points in camera coordinates
      K  : (3,3) intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
      D  : (4,) distortion coefficients [k1, k2, k3, k4]
    
    Output:
      uv : (2,) or (N,2) array of pixel coordinates   

    Mathematical Model:
    -------------------
    1. Spherical coordinates:
       r = sqrt(X² + Y²)
       θ = atan(r / Z)
    
    2. Radial distortion:
       θ_d = θ * (1 + k₁*θ² + k₂*θ⁴ + k₃*θ⁶ + k₄*θ⁸)
    
    3. Normalized image coordinates:
       x' = (θ_d / r) * X
       y' = (θ_d / r) * Y
    
    4. Pixel coordinates:
       u = fx * x' + cx
       v = fy * y' + cy
    """
    Pc = np.asarray(Pc, dtype=float)
    single_point = False
    if Pc.ndim == 1:
        Pc = Pc.reshape(1, 3)
        single_point = True

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    k1, k2, k3, k4 = D[:4]

    X = Pc[:, 0]
    Y = Pc[:, 1]
    Z = Pc[:, 2]

    # Avoid division by zero
    Z = np.where(Z == 0, 1e-9, Z)

    # Normalized pinhole coordinates
    a = X / Z
    b = Y / Z

    r = np.sqrt(a**2 + b**2)

    # Handle points on the optical axis separately
    theta = np.arctan(r)
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)

    # When r == 0, set safe scale = 1 (point projects to principal point)
    scale = np.ones_like(r)
    nonzero = r > 1e-12
    scale[nonzero] = theta_d[nonzero] / r[nonzero]

    x_prime = scale * a
    y_prime = scale * b

    u = fx * x_prime + cx
    v = fy * y_prime + cy

    uv = np.stack((u, v), axis=-1)

    if single_point:
        return uv[0]
    return uv

def _newton_solve_theta(rd, k1, k2, k3, k4, max_iter=10, eps=1e-9):
    """
    Solve for theta in:
        rd = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    using Newton-Raphson.    
    rd can be scalar or array.
    """
    rd = np.asarray(rd, dtype=float)
    theta = rd.copy()  # good initial guess

    for _ in range(max_iter):
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta4 * theta2
        theta8 = theta4**2

        # f(theta)
        poly = 1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8
        f = theta * poly - rd

        # f'(theta)
        dpoly = 1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8

        # Avoid division by zero
        dpoly = np.where(np.abs(dpoly) < 1e-12, 1e-12, dpoly)

        delta = f / dpoly
        theta = theta - delta

        if np.all(np.abs(delta) < eps):
            break

    return theta


def unprojectKannalaBrandt(uv, K, D, max_iter=10, eps=1e-9, normalize=True):
    """
    Unproject pixel coordinates to 3D rays using the Kannala-Brandt fisheye model.
    
    Inputs:
      uv        : (2,) or (N,2) pixel coordinates
      K         : (3,3) intrinsic matrix
      D         : (4,) distortion coefficients [k1,k2,k3,k4]
      max_iter  : max Newton iterations
      eps       : tolerance for Newton iterations
      normalize : if True, return unit-length rays. If False, return [x, y, 1].
    
    Output:
      rays : (3,) or (N,3) array of 3D rays in camera coordinates
    """
    uv = np.asarray(uv, dtype=float)
    single_point = False
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
        single_point = True

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    k1, k2, k3, k4 = D[:4]

    u = uv[:, 0]
    v = uv[:, 1]

    # Distorted normalized coordinates
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    rd = np.sqrt(x_d**2 + y_d**2)

    # Handle rd == 0 (principal point): ray is along optical axis
    rays = np.zeros((uv.shape[0], 3), dtype=float)
    on_axis = rd < 1e-12
    if np.any(on_axis):
        rays[on_axis, :] = np.array([0.0, 0.0, 1.0])

    # For others, solve for theta and compute undistorted coords
    off_axis = ~on_axis
    if np.any(off_axis):
        rd_ = rd[off_axis]
        x_d_ = x_d[off_axis]
        y_d_ = y_d[off_axis]

        theta = _newton_solve_theta(rd_, k1, k2, k3, k4,
                                    max_iter=max_iter, eps=eps)

        r = np.tan(theta)
        # Avoid division by zero
        rd_safe = np.where(rd_ < 1e-12, 1e-12, rd_)
        s = r / rd_safe

        a = s * x_d_
        b = s * y_d_

        # Construct ray [a, b, 1]
        rays[off_axis, 0] = a
        rays[off_axis, 1] = b
        rays[off_axis, 2] = 1.0

    if normalize:
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        rays = rays / norms

    if single_point:
        return rays[0]
    return rays

def testingKannalaBrandt():
    # ---------------------------------------------------------
    # 1) Load calibration
    # ---------------------------------------------------------
    K = np.loadtxt(basePath + "K_1.txt")
    D_all = np.loadtxt(basePath + "D1_k_array.txt")
    D = D_all[:4]   # we use only the fisheye 4-coefficient model

    # ---------------------------------------------------------
    # 2) Virtual 3D points X_i (we use only first 3 components)
    #    These are in camera coordinates for the test
    # ---------------------------------------------------------
    X = np.array([
        [ 3.0,  2.0, 10.0],   # X1
        [-5.0,  6.0,  7.0],   # X2
        [ 1.0,  5.0, 14.0],   # X3
    ])

    # Corresponding given image points u_i (ignore last component = 1)
    U_gt = np.array([
        [503.387,  450.1594],   # u1
        [267.9465, 580.4671],   # u2
        [441.0609, 493.0671],   # u3
    ])

    print("K =\n", K)
    print("D =", D)

    # ---------------------------------------------------------
    # 3) PROJECTION TEST: X_i -> u_pred
    # ---------------------------------------------------------
    u_pred = projectKannalaBrandt(X, K, D) 

    print("\n=== Given pixels (ground truth) ===")
    print(U_gt)

    print("\n=== Projected pixels (our KB implementation) ===")
    print(u_pred)

    pixel_errors = np.linalg.norm(u_pred - U_gt, axis=1)
    print("\n=== Pixel reprojection error per point (in pixels) ===")
    for i, e in enumerate(pixel_errors):
        print(f"Point X{i+1}: error = {e:.6f} px")

    # ---------------------------------------------------------
    # 4) UNPROJECTION TEST: u_i -> ray_i, compare with X_i
    # ---------------------------------------------------------
    rays = unprojectKannalaBrandt(U_gt, K, D, normalize=True)

    # Normalize original X to directions for fair comparison
    X_dir = X / np.linalg.norm(X, axis=1, keepdims=True)

    print("\n=== Unprojected rays (unit length) ===")
    print(rays)

    print("\n=== Angular error between original X direction and unprojected ray ===")
    for i in range(X.shape[0]):
        dot = np.clip(np.dot(X_dir[i], rays[i]), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        print(f"Point {i+1}: {angle_deg:.8f} degrees")


if __name__ == "__main__":
    print("Kannala-Brandt Camera Model")
    testingKannalaBrandt()
    print("=" * 50)
    print()
    print("This module implements projection and unprojection for fisheye cameras.")
