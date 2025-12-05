#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 5
#
# Title: Laboratory 5: Bundle Adjustment using calibrated stereo with fish-eyes (optional).
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################

# ############################################################################
# BUNDLE ADJUSTMENT - CALIBRATED STEREO WITH FISHEYE (LAB 5)
# ############################################################################

import numpy as np
import scipy.optimize as scOptim
from kannalaBrandt import (projectKannalaBrandt, 
                            unprojectKannalaBrandt)

basePath = "labSession5/"

def resBundleFisheyeStereo(Op, x1Data, x2Data, K1, D1, K2, D2, T_wc1, T_wc2):
    """
    Residual function for bundle adjustment with calibrated stereo fisheye.

    Unknowns:
      Op      : (3*nPoints,) flattened 3D points in WORLD frame

    Known:
      x1Data  : (3xnPoints) observed pixels in cam1 (homogeneous)
      x2Data  : (3xnPoints) observed pixels in cam2 (homogeneous)
      K1, D1  : intrinsics + distortion for cam1
      K2, D2  : intrinsics + distortion for cam2
      T_wc1   : (4x4) camera 1 pose in world frame
      T_wc2   : (4x4) camera 2 pose in world frame

    Output:
      res     : (4*nPoints,) residual vector:
                [res_x1, res_y1, res_x2, res_y2] for each point
    """
    x1Data = np.asarray(x1Data, dtype=float)
    x2Data = np.asarray(x2Data, dtype=float)
    nPoints = x1Data.shape[1]

    # 1) Recover 3D points in WORLD frame
    Xw = Op.reshape(3, nPoints)  # 3 x N

    # 2) Precompute world->camera transforms
    T_wc1 = np.asarray(T_wc1, dtype=float).reshape(4, 4)
    T_wc2 = np.asarray(T_wc2, dtype=float).reshape(4, 4)

    T_cw1 = np.linalg.inv(T_wc1)
    T_cw2 = np.linalg.inv(T_wc2)

    R_cw1 = T_cw1[:3, :3]
    t_cw1 = T_cw1[:3, 3]
    R_cw2 = T_cw2[:3, :3]
    t_cw2 = T_cw2[:3, 3]

    res = np.empty(4 * nPoints)

    for i in range(nPoints):
        X = Xw[:, i]

        # cam1: Xc1 = R_cw1 * Xw + t_cw1
        Xc1 = R_cw1 @ X + t_cw1
        u1_pred, v1_pred = projectKannalaBrandt(Xc1, K1, D1)

        # cam2: Xc2 = R_cw2 * Xw + t_cw2
        Xc2 = R_cw2 @ X + t_cw2
        u2_pred, v2_pred = projectKannalaBrandt(Xc2, K2, D2)

        u1_obs = x1Data[0, i]
        v1_obs = x1Data[1, i]
        u2_obs = x2Data[0, i]
        v2_obs = x2Data[1, i]

        # Follow the same sign convention as resBundleProjection: obs - pred
        res[4*i]     = u1_obs - u1_pred
        res[4*i + 1] = v1_obs - v1_pred
        res[4*i + 2] = u2_obs - u2_pred
        res[4*i + 3] = v2_obs - v2_pred

    return res



def bundleAdjustmentFisheyeStereo(x1Data, x2Data,
                                  K1, D1, K2, D2,
                                  T_wc1, T_wc2,
                                  Xw_init):
    """
    Perform bundle adjustment for calibrated stereo fisheye (Lab 5).

    -input:
      x1Data : (3xN) measured pixels in cam1 (homogeneous)
      x2Data : (3xN) measured pixels in cam2 (homogeneous)
      K1,D1  : intrinsics + distortion for cam1
      K2,D2  : intrinsics + distortion for cam2
      T_wc1  : (4x4) cam1 pose in world
      T_wc2  : (4x4) cam2 pose in world
      Xw_init: (3xN) initial 3D points in world frame (e.g. from triangulation)

    -output:
      Xw_opt     : (3xN) optimized 3D points
      res_initial: initial residuals
      res_final  : final residuals
    """
    nPoints = x1Data.shape[1]

    # Flatten initial 3D points
    Op_init = Xw_init.flatten()

    # Initial residuals
    res_initial = resBundleFisheyeStereo(Op_init, x1Data, x2Data,
                                         K1, D1, K2, D2,
                                         T_wc1, T_wc2)
    cost_initial = np.sum(res_initial**2)
    rms_initial = np.sqrt(cost_initial / (4 * nPoints))

    print("=== Fisheye Stereo BA - INITIAL STATE ===")
    print(f"Initial cost: {cost_initial:.6f}")
    print(f"RMS reprojection error: {rms_initial:.6f} pixels")
    print()

    # Optimize only the 3D points
    result = scOptim.least_squares(
        resBundleFisheyeStereo,
        Op_init,
        args=(x1Data, x2Data, K1, D1, K2, D2, T_wc1, T_wc2),
        method='lm',
        verbose=2
    )

    Op_opt = result.x
    res_final = result.fun
    cost_final = result.cost
    rms_final = np.sqrt(2 * cost_final / (4 * nPoints))  # same style as your code

    print()
    print("=== Fisheye Stereo BA - RESULT ===")
    print(f"Final cost: {cost_final:.6f}")
    print(f"RMS reprojection error: {rms_final:.6f} pixels")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nfev}")
    print()

    Xw_opt = Op_opt.reshape(3, nPoints)

    return Xw_opt, res_initial, res_final


if __name__ == "__main__":
    # 1) Load 2D data and initial 3D
    X1 = np.loadtxt(basePath + "x1.txt")   # 3 x N
    X2 = np.loadtxt(basePath + "x2.txt")   # 3 x N
    x1Data = X1                 # keep as 3xN homogeneous
    x2Data = X2

    Xw_init = np.loadtxt("points3D_poseA.txt")  # (N,3)
    Xw_init = Xw_init.T                         # make it 3xN

    K1 = np.loadtxt(basePath + "K_1.txt")
    K2 = np.loadtxt(basePath + "K_2.txt")
    D1_all = np.loadtxt(basePath + "D1_k_array.txt")
    D2_all = np.loadtxt(basePath + "D2_k_array.txt")
    D1 = D1_all[:4]
    D2 = D2_all[:4]

    T_wc1 = np.loadtxt(basePath + "T_wc1.txt")
    T_wc2 = np.loadtxt(basePath + "T_wc2.txt")

    # 2) Run BA
    Xw_opt, res_initial, res_final = bundleAdjustmentFisheyeStereo(
        x1Data, x2Data,
        K1, D1, K2, D2,
        T_wc1, T_wc2,
        Xw_init
    )

    # 3) Save optimized 3D
    np.savetxt("points3D_poseA_BA.txt", Xw_opt.T)
    print("Saved BA-refined 3D points to points3D_poseA_BA.txt")
