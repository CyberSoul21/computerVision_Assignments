#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4 - Section 4
#
# Title: 4)Bundle Adjustment Generalized N-View 
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################

import numpy as np
import cv2

from helperFunctions import *
from labSession4.plotGroundTruth import *

basePath = "labSession4/"
outPutPath = "output/"

def visualizeNViewResults(xData_list, T_scaled_list, X1_scaled, K_c, T_GT_list, X1_GT, image_paths=None):
    """
    Visualize N-view bundle adjustment results.
    
    -input:
      xData_list: List of observed 2D points per camera
      T_scaled_list: List of optimized camera poses (scaled)
      X1_scaled: Optimized 3D points (scaled)
      K_c: Camera calibration matrix
      T_GT_list: Ground truth camera poses (for comparison)
      image_paths: Optional list of image file paths
    """
    
    nCameras = len(xData_list)
    nPoints = X1_scaled.shape[1]
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80 + "\n")
    
    # ########################################
    # 1. REPROJECTIONS IN ALL IMAGES
    # ########################################
    
    print("Generating reprojection visualizations...")
    
    # Determine subplot layout
    if nCameras <= 3:
        fig, axes = plt.subplots(1, nCameras, figsize=(6*nCameras, 6))
    elif nCameras == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        ncols = 3
        nrows = (nCameras + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        axes = axes.flatten()
    
    # Make axes iterable even for single camera
    if nCameras == 1:
        axes = [axes]
    
    # Project and plot for each camera
    X1_hom = np.vstack((X1_scaled, np.ones((1, nPoints))))
    
    for cam_idx in range(nCameras):
        ax = axes[cam_idx]
        
        # Load image if path provided
        if image_paths and cam_idx < len(image_paths):
            img = cv2.imread(image_paths[cam_idx])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                h, w = img.shape[:2]
                ax.set_xlim([0, w])
                ax.set_ylim([h, 0])
        
        # Project 3D points to this camera
        T_cam = T_scaled_list[cam_idx][:3, :]
        x_proj = K_c @ T_cam @ X1_hom
        x_proj = x_proj / x_proj[2, :]
        
        # Plot residuals and points
        xData = xData_list[cam_idx]
        plotResidual(xData, x_proj, 'y-')
        ax.plot(xData[0, :], xData[1, :], 'rx', 
                label='Observed', markersize=8, markeredgewidth=2)
        ax.plot(x_proj[0, :], x_proj[1, :], 'go', 
                label='Projected', markersize=6, fillstyle='none', markeredgewidth=2)
        
        ax.set_title(f'Camera {cam_idx+1} - Reprojection', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    
    # Hide unused subplots
    for i in range(nCameras, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(outPutPath + f'04_final_reprojections_{nCameras}views.png', dpi=150, bbox_inches='tight')
    print(f"* Saved: final_reprojections_{nCameras}views.png")
    plt.show()
    
    # ########################################
    # 2. 3D VISUALIZATION
    # ########################################
    
    print("Generating 3D visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate appropriate scale for camera axes
    # (drawRefSystem uses unit vectors, we want them proportional to scene)
    points_range = np.array([
        X1_scaled[0, :].max() - X1_scaled[0, :].min(),
        X1_scaled[1, :].max() - X1_scaled[1, :].min(),
        X1_scaled[2, :].max() - X1_scaled[2, :].min()
    ])
    max_range = points_range.max()
    axis_scale = max_range / 3.0  # Camera axes will be 1/3 of scene size
    
    # Helper function to scale camera coordinate system
    def scaleTransform(T, scale):
        """Scale the rotation axes of a transformation matrix."""
        T_scaled = T.copy()
        T_scaled[:3, 0:3] *= scale  # Scale rotation axes only
        return T_scaled
    
    # Draw C1 at origin (reference frame)
    # Note: C1_GT is also at origin, so they overlap
    T1_display = scaleTransform(T_scaled_list[0], axis_scale)
    drawRefSystem(ax, T1_display, '-', 'C1 (Ref)')
    
    # Draw other estimated cameras (solid lines)
    for cam_idx in range(1, nCameras):
        T_display = scaleTransform(T_scaled_list[cam_idx], axis_scale)
        drawRefSystem(ax, T_display, '-', f'C{cam_idx+1}')
    
    # Draw ground truth cameras (dashed lines)
    # Skip C1_GT since it overlaps with C1 at origin
    if T_GT_list is not None:
        for cam_idx in range(1, min(len(T_GT_list), nCameras)):
            T_GT_display = scaleTransform(T_GT_list[cam_idx], axis_scale)
            drawRefSystem(ax, T_GT_display, '--', f'C{cam_idx+1}_GT')
    
    # Plot 3D points
    ax.scatter(X1_scaled[0, :], X1_scaled[1, :], X1_scaled[2, :], 
               c='red', marker='o', s=20, label='3D Points (BA)', alpha=0.6)
    ax.scatter(X1_GT[0, :], X1_GT[1, :], X1_GT[2, :], 
               c='blue', marker='x', s=20, label='3D Points (GT)', alpha=0.6)
    
    # Calculate axis limits including cameras
    all_coords = []
    for T in T_scaled_list:
        all_coords.append(T[:3, 3])
    if T_GT_list is not None:
        for T in T_GT_list:
            all_coords.append(T[:3, 3])
    
    all_coords = np.array(all_coords).T if all_coords else np.zeros((3, 1))
    
    # Combine points and camera positions
    min_x = min(X1_scaled[0, :].min(), all_coords[0, :].min() if all_coords.size > 0 else 0)
    max_x = max(X1_scaled[0, :].max(), all_coords[0, :].max() if all_coords.size > 0 else 0)
    min_y = min(X1_scaled[1, :].min(), all_coords[1, :].min() if all_coords.size > 0 else 0)
    max_y = max(X1_scaled[1, :].max(), all_coords[1, :].max() if all_coords.size > 0 else 0)
    min_z = min(X1_scaled[2, :].min(), all_coords[2, :].min() if all_coords.size > 0 else 0)
    max_z = max(X1_scaled[2, :].max(), all_coords[2, :].max() if all_coords.size > 0 else 0)
    
    # Add margin
    margin = max_range * 0.15
    
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_zlim(min_z - margin, max_z + margin)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title(f'Bundle Adjustment - {nCameras} Cameras (Camera 1 Frame)', 
                 fontsize=14, fontweight='bold')
    
    # Add note about overlapping cameras
    note_text = "Note: C1 and C1_GT overlap at origin"
    ax.text2D(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.savefig(outPutPath + f'04_final_3d_{nCameras}views.png', dpi=150, bbox_inches='tight')
    print(f"* Saved: 04_final_3d_{nCameras}views.png")
    plt.show()



# ########################################
# MAIN PIPELINE
# ########################################

if __name__ == '__main__':
    np.set_printoptions(precision=6, linewidth=1024, suppress=True)
    
    print("\n" + "=" * 80)
    print(" " * 10 + "LABORATORY 4 - SECTION 4: 3-VIEW BUNDLE ADJUSTMENT")
    print("=" * 80 + "\n")
    print()
    
    # ########################################
    # STEP 1: Load all data
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
    
    nPoints = x1Data.shape[1]
    print(f"* Loaded {nPoints} point correspondences across 3 views")
    print()
    
    # ########################################
    # STEP 2: Initial pose estimation (C1 & C2)
    # ########################################
    
    print("STEP 2: Initial pose estimation (cameras 1 & 2)...")
    print("-" * 80)
    
    E = essentialMatrix(F_21, K_c)
    solutions = decomposeEssentialMatrix(E)
    x1_2d = x1Data[:2, :].T
    x2_2d = x2Data[:2, :].T
    (R_21, t_21), X1_init = selectCorrectPose(K_c, x1_2d, x2_2d, solutions)
    
    T_21_init = np.eye(4)
    T_21_init[:3, :3] = R_21
    T_21_init[:3, 3] = t_21
    X1_init = X1_init.T  # Convert to 3xN
    
    print("* Initial pose computed from Essential matrix")
    print()
    
    # ########################################
    # STEP 3: Bundle Adjustment (2 views) using GENERALIZED function
    # ########################################
    
    print("STEP 3: Bundle Adjustment (cameras 1 & 2) - Using N-view BA with N=2")
    print("-" * 80)
    print()
    
    # Prepare data for 2-view BA using the SAME generalized function!
    xData_list_2views = [x1Data, x2Data]
    T_1 = np.eye(4)
    T_init_list_2views = [T_1, T_21_init]
    
    # Call the generalized function with N=2
    T_opt_list_2views, X1_refined, stats_2views = bundleAdjustmentNViews(
        xData_list_2views, K_c, T_init_list_2views, X1_init
    )
    
    print(f"* 2-view BA completed using generalized N-view function")
    print(f"  Final RMS: {stats_2views['rms_final']:.6f} pixels")
    print()
    
    # Extract refined pose and points
    T_21_refined = T_opt_list_2views[1]
    
    # ########################################
    # STEP 4: PnP for Camera 3
    # ########################################
    
    print("STEP 4: PnP pose estimation (camera 3)...")
    print("-" * 80)
    
    objectPoints = X1_refined.T  # Nx3
    imagePoints = np.ascontiguousarray(x3Data[0:2, :].T).reshape((nPoints, 1, 2))
    
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
    
    R_31, _ = cv2.Rodrigues(rvec)
    T_31_init = np.eye(4)
    T_31_init[:3, :3] = R_31
    T_31_init[:3, 3] = tvec.flatten()
    
    print(f"* Camera 3 pose estimated")
    print()
    
    # ########################################
    # STEP 5: Bundle Adjustment (3 views) using SAME generalized function
    # ########################################
    
    print("STEP 5: Bundle Adjustment (ALL 3 cameras) - Using N-view BA with N=3")
    print("-" * 80)
    print()

    
    # Prepare data for 3-view BA using the SAME generalized function!
    xData_list_3views = [x1Data, x2Data, x3Data]
    T_init_list_3views = [T_1, T_21_refined, T_31_init]
    
    # Call the SAME generalized function with N=3
    T_opt_list_3views, X1_final, stats_3views = bundleAdjustmentNViews(
        xData_list_3views, K_c, T_init_list_3views, X1_refined
    )
    
    print(f"* 3-view BA completed using generalized N-view function")
    print(f"  Final RMS: {stats_3views['rms_final']:.6f} pixels")
    print()
    
    # ########################################
    # STEP 6: Fix scale
    # ########################################
    
    # Transform ground truth to camera 1 frame   
    T_21_GT = np.linalg.inv(T_wc1_GT) @ T_wc2_GT
    T_31_GT = np.linalg.inv(T_wc1_GT) @ T_wc3_GT
    
    T_GT_list = [np.eye(4), T_21_GT, T_31_GT]
    
    T_scaled_list, X1_scaled, scale_factor = fixScaleNViews(
        T_opt_list_3views, X1_final, T_GT_list
    )
    
    # ########################################
    # STEP 7: Compare with ground truth
    # ########################################
    
    print("=" * 80)
    print("COMPARISON WITH GROUND TRUTH")
    print("=" * 80)
    
    t_error_2 = np.linalg.norm(T_scaled_list[1][:3, 3] - T_21_GT[:3, 3])
    t_error_3 = np.linalg.norm(T_scaled_list[2][:3, 3] - T_31_GT[:3, 3])
    
    R_error_2 = np.linalg.norm(T_scaled_list[1][:3, :3] - T_21_GT[:3, :3], 'fro')
    R_error_3 = np.linalg.norm(T_scaled_list[2][:3, :3] - T_31_GT[:3, :3], 'fro')
    
    print(f"\nCamera 2:")
    print(f"  Translation error: {t_error_2:.6f} units")
    print(f"  Rotation error:    {R_error_2:.6f} (Frobenius norm)")
    
    print(f"\nCamera 3:")
    print(f"  Translation error: {t_error_3:.6f} units")
    print(f"  Rotation error:    {R_error_3:.6f} (Frobenius norm)")
    print()
    
    # Ground truth for comparison (transform to camera 1 frame)
    T_12_GT = np.linalg.inv(T_wc1_GT) @ T_wc2_GT
    X1_GT = np.linalg.inv(T_wc1_GT) @ X_w_GT
    X1_GT = X1_GT[:3, :] / X1_GT[3, :]

    # ########################################
    # STEP 8: Visualize results
    # ########################################
    
    image_paths = [basePath + 'image1.png', basePath + 'image2.png', basePath + 'image3.png']
    
    visualizeNViewResults(
        xData_list_3views, T_scaled_list, X1_scaled, K_c, T_GT_list, X1_GT, image_paths
    )
    
    # ########################################
    # FINAL RESULTS
    # ########################################
    
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL RESULTS")
    print("=" * 80)
    print()
    print("2-View Bundle Adjustment:")
    print(f"  Initial RMS error:       {stats_2views['rms_initial']:.6f} pixels")
    print(f"  Final RMS error:         {stats_2views['rms_final']:.6f} pixels")
    print(f"  Improvement:             {stats_2views['improvement_pct']:.2f}%")
    print()
    print("3-View Bundle Adjustment:")
    print(f"  Initial RMS error:       {stats_3views['rms_initial']:.6f} pixels")
    print(f"  Final RMS error:         {stats_3views['rms_final']:.6f} pixels")
    print(f"  Improvement:             {stats_3views['improvement_pct']:.2f}%")
    print()
    print("Scale Correction:")
    print(f"  Scale factor applied:    {scale_factor:.6f}")
    print()
    print("Accuracy vs Ground Truth:")
    print(f"  Camera 2 pose error:     {t_error_2:.6f} units")
    print(f"  Camera 3 pose error:     {t_error_3:.6f} units")
    print()
