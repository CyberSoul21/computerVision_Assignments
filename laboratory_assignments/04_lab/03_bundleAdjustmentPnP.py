#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: 3) Complete Pipeline - Bundle Adjustment (2 views) + PnP (camera 3)
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.linalg as scAlg
import scipy.optimize as scOptim
from mpl_toolkits.mplot3d import Axes3D

from helperFunctions import *
from labSession4.plotGroundTruth import *

basePath = "labSession4/"
outPutPath = "output/"

# ############################################################################
# MAIN PIPELINE
# ############################################################################

if __name__ == '__main__':
    np.set_printoptions(precision=6, linewidth=1024, suppress=True)
    
    print("\n" + "=" * 80)
    print(" " * 20 + "COMPLETE PIPELINE: 3 CAMERAS")
    print("=" * 80 + "\n")
    
    # ########################################
    # STEP 1: Load data
    # ########################################
    
    print("STEP 1: Loading data...")
    print("-" * 80)
    
    x1Data = np.loadtxt(basePath + 'x1Data.txt')
    x2Data = np.loadtxt(basePath + 'x2Data.txt')
    x3Data = np.loadtxt(basePath + 'x3Data.txt')
    K_c = np.loadtxt(basePath + 'K_c.txt')
    F_21 = np.loadtxt(basePath + 'F_21.txt')
    
    # Load ground truth for comparison
    T_wc1_GT = np.loadtxt(basePath + 'T_w_c1.txt')
    T_wc2_GT = np.loadtxt(basePath + 'T_w_c2.txt')
    T_wc3_GT = np.loadtxt(basePath + 'T_w_c3.txt')
    X_w_GT = np.loadtxt(basePath + 'X_w.txt')
    
    print(f"* Loaded {x1Data.shape[1]} point correspondences")
    print()
    
    # ########################################
    # STEP 2: Initial pose from Essential Matrix
    # ########################################
    
    print("STEP 2: Computing initial pose (cameras 1 & 2)...")
    print("-" * 80)
    
    E = essentialMatrix(F_21, K_c)
    solutions = decomposeEssentialMatrix(E)

    x1_2d = x1Data[:2, :].T
    x2_2d = x2Data[:2, :].T
    (R_21, t_21), X1_init = selectCorrectPose(K_c, x1_2d, x2_2d, solutions) #From triangulation
    
    T_21_init = np.eye(4)
    T_21_init[:3, :3] = R_21
    T_21_init[:3, 3] = t_21
    X1_init = X1_init.T
    
    print("* Initial pose computed")
    print()
    
    # ########################################
    # STEP 3: Bundle Adjustment (2 views)
    # ########################################

    ##para escalar al GT en METROS!!!
    T_c1_w_GT = np.linalg.inv(T_wc1_GT)
    T_c2_w_GT = np.linalg.inv(T_wc2_GT)    
    T12_gt = T_c1_w_GT @ T_wc2_GT
    t12 = T12_gt[0:3, 3]
    normal_t12 = np.linalg.norm(t12)
    print("t12 norm:", np.linalg.norm(t12)) #sanity check 
    #######  
    
    print("STEP 3: Bundle Adjustment (cameras 1 & 2)...")
    print("-" * 80)
    
    T_21_opt, X1_opt, _, _ = bundleAdjustment(x1Data, x2Data, K_c, T_21_init, X1_init, normal_t12)
    
    print("* Bundle Adjustment completed")
    print()
    
    # ########################################
    # STEP 4: PnP for Camera 3
    # ########################################
    
    print("STEP 4: PnP Pose Estimation (camera 3)...")
    print("-" * 80)
    
    # Prepare data for PnP
    nPoints = X1_opt.shape[1]
    objectPoints = X1_opt.T  # Nx3
    imagePoints = np.ascontiguousarray(x3Data[0:2, :].T).reshape((nPoints, 1, 2))
    
    # Solve PnP
    retval, rvec, tvec = cv2.solvePnP(
        objectPoints,
        imagePoints,
        K_c,
        None,  # No distortion
        flags=cv2.SOLVEPNP_EPNP
    )
    
    if not retval:
        print("ERROR: PnP failed!")
        exit(1)
    
    # Convert to transformation matrix
    R_31, _ = cv2.Rodrigues(rvec)
    T_31 = np.eye(4)
    T_31[:3, :3] = R_31
    T_31[:3, 3] = tvec.flatten()
    
    print("* Camera 3 pose estimated")
    print(f"  Translation: {tvec.flatten()}")
    print()
    
    # ########################################
    # STEP 5: Validate and visualize
    # ########################################
    
    print("STEP 5: Validation and Visualization...")
    print("-" * 80)
    
    # Project to camera 3
    X1_hom = np.vstack((X1_opt, np.ones((1, nPoints))))
    x3_proj = K_c @ T_31[:3, :] @ X1_hom
    x3_proj = x3_proj / x3_proj[2, :]
    
    # Compute error
    res_x = x3Data[0, :] - x3_proj[0, :]
    res_y = x3Data[1, :] - x3_proj[1, :]
    errors = np.sqrt(res_x**2 + res_y**2)
    rms_error = np.sqrt(np.mean(errors**2))
    
    print(f"Reprojection RMS error (camera 3): {rms_error:.4f} pixels")
    print()
    
    # Compare with ground truth
    T_31_GT = np.linalg.inv(T_wc1_GT) @ T_wc3_GT
    t_error = np.linalg.norm(T_31[:3, 3] - T_31_GT[:3, 3])

    X1_GT = np.linalg.inv(T_wc1_GT) @ X_w_GT
    X1_GT = X1_GT[:3, :] / X1_GT[3, :]
    
    print(f"Comparison with Ground Truth:")
    print(f"  Translation error: {t_error:.6f} units")
    print()
    
    # ########################################
    # VISUALIZATION
    # ########################################
    
    # 1. Reprojection in image 3
    image3 = cv2.imread(basePath + 'image3.png')
    if image3 is not None:
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if image3 is not None:
        ax.imshow(image3)
        h, w = image3.shape[:2]
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
    
    plotResidual(x3Data, x3_proj, 'y-')
    ax.plot(x3Data[0, :], x3Data[1, :], 'rx', 
            label='Observed', markersize=10, markeredgewidth=2)
    ax.plot(x3_proj[0, :], x3_proj[1, :], 'co', 
            label='Projected (PnP)', markersize=8, fillstyle='none', markeredgewidth=2)
    ax.set_title('Camera 3 - PnP Result', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(outPutPath + '03_step3_pnp_reprojection.png', dpi=150, bbox_inches='tight')
    print("Saved: 03_step3_pnp_reprojection.png")
    plt.show()
    
    # 2. 3D visualization with all 3 cameras
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera 1 (origin), Camera 1 Reference Frame
    T_1 = np.eye(4)
    drawRefSystem(ax, T_1, '-', 'C1')
    
    # Camera 2 (from bundle adjustment)
    drawRefSystem(ax, T_21_opt, '-', 'C2')
    
    # Camera 3 (from PnP)
    drawRefSystem(ax, T_31, '-', 'C3 (PnP)')
    
    # Camera 3 ground truth
    drawRefSystem(ax, T_31_GT, '--', 'C3 (GT)')
   
    # 3D points
    ax.scatter(X1_opt[0, :], X1_opt[1, :], X1_opt[2, :], 
               c='red', marker='o', s=30, label='3D Points', alpha=0.6)
    ax.scatter(X1_GT[0, :], X1_GT[1, :], X1_GT[2, :], 
               c='blue', marker='x', s=20, label='3D Points (GT)', alpha=0.6)    
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('3D Reconstruction - 3 Cameras', fontsize=14, fontweight='bold')
    
    plt.savefig(outPutPath + '03_step3_all_cameras_3d.png', dpi=150, bbox_inches='tight')
    print("* Saved: 03_step3_all_cameras_3d.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("\nResults:")
    print(f"  * Cameras 1 & 2: Bundle Adjustment completed")
    print(f"  * Camera 3: PnP estimation completed")
    print(f"  * Reprojection error: {rms_error:.4f} pixels")
    print(f"  * Pose error vs ground truth: {t_error:.6f} units")
    print("=" * 80 + "\n")