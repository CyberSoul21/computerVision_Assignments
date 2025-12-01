#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: 2) Bundle Adjustment from Two Views
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
# MAIN
# ############################################################################


def plot2DPoints(image1,image2,x1Data,x2Data,x1_proj_init,x2_proj_init,labelMessage,title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Image 1
    if image1 is not None:
        axes[0].imshow(image1)
        h, w = image1.shape[:2]
        axes[0].set_xlim([0, w])
        axes[0].set_ylim([h, 0])  # Invert Y axis for image coordinates
    
    #plotResidual(x1Data, x1_proj_init, 'y-')
    for k in range(x1Data.shape[1]):
        axes[0].plot([x1Data[0, k], x1_proj_init[0, k]], [x1Data[1, k], x1_proj_init[1, k]], 'k-')
    axes[0].plot(x1Data[0, :], x1Data[1, :], 'rx', label='Observed', markersize=10, markeredgewidth=2)
    axes[0].plot(x1_proj_init[0, :], x1_proj_init[1, :], 'bo', 
                 label=labelMessage, markersize=8, fillstyle='none', markeredgewidth=2)
    axes[0].set_title('Image 1 - '+title, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    # Image 2
    if image2 is not None:
        axes[1].imshow(image2)
        h, w = image2.shape[:2]
        axes[1].set_xlim([0, w])
        axes[1].set_ylim([h, 0])  # Invert Y axis for image coordinates
    
    #plotResidual(x2Data, x2_proj_init, 'y-')
    for k in range(x2Data.shape[1]):
        axes[1].plot([x2Data[0, k], x2_proj_init[0, k]], [x2Data[1, k], x2_proj_init[1, k]], 'k-')
    axes[1].plot(x2Data[0, :], x2Data[1, :], 'rx', label='Observed', markersize=10, markeredgewidth=2)
    axes[1].plot(x2_proj_init[0, :], x2_proj_init[1, :], 'bo', 
                 label=labelMessage, markersize=8, fillstyle='none', markeredgewidth=2)
    axes[1].set_title('Image 2 - '+title, fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(outPutPath + '02_residuals_initial.png', dpi=150, bbox_inches='tight')
    print("* Saved: 02_residuals_initial.png")
    plt.show()



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    
    print("=" * 70)
    print("BUNDLE ADJUSTMENT TEST - TWO VIEWS")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    x1Data = np.loadtxt(basePath + 'x1Data.txt') #contains the 2D points (pixel coordinates) observed in image 1, which is captured by Camera 1.
    x2Data = np.loadtxt(basePath + 'x2Data.txt') #contains the 2D points (pixel coordinates) observed in image 2, which is captured by Camera 2.
    K_c = np.loadtxt(basePath + 'K_c.txt')
    F_21 = np.loadtxt(basePath + 'F_21.txt')
    
    # Load ground truth for comparison
    T_wc1_GT = np.loadtxt(basePath + 'T_w_c1.txt')
    T_wc2_GT = np.loadtxt(basePath + 'T_w_c2.txt')
    X_w_GT = np.loadtxt(basePath + 'X_w.txt') #contains the ground truth 3D points in the world coordinate frame.
    
    print(f"Loaded {x1Data.shape[1]} point correspondences")
    print()


    # Load images
    image1 = cv2.imread(basePath + 'image1.png')
    image2 = cv2.imread(basePath + 'image2.png')    
    if image1 is None:
        print("WARNING: Could not load image1.png")
    else:
        print(f"* Loaded image1.png - Shape: {image1.shape}")
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    if image2 is None:
        print("WARNING: Could not load image2.png")
    else:
        print(f"* Loaded image2.png - Shape: {image2.shape}")
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # ########################################
    # STEP 1: Initial solution from Essential Matrix
    # ########################################
    
    print("STEP 1: Computing initial pose from Essential Matrix...")
    print("-" * 70)
    
    E = essentialMatrix(F_21, K_c)
    solutions = decomposeEssentialMatrix(E)
    
    # Convert 2D points to proper format for triangulation
    x1_2d = x1Data[:2, :].T  # Nx2
    x2_2d = x2Data[:2, :].T  # Nx2
    
    (R_21, t_21), X1_init = selectCorrectPose(K_c, x1_2d, x2_2d, solutions) # Form Triagulation
    
    # Convert to homogeneous transformation
    T_21_init = np.eye(4)
    T_21_init[:3, :3] = R_21
    T_21_init[:3, 3] = t_21
    
    # Convert points to 3xN format
    X1_init = X1_init.T  # 3xN
    
    print(f"Initial rotation:\n{R_21}")
    print(f"Initial translation: {t_21}")
    print()
    
    # ########################################
    # STEP 2: Visualize initial residuals
    # ########################################
    
    print("STEP 2: Visualizing initial residuals...")
    print("-" * 70) 

    
    # Project initial 3D points, from triangulation.
    X1_hom = np.vstack((X1_init, np.ones((1, X1_init.shape[1]))))

    x1_proj_init = K_c @ np.eye(3, 4) @ X1_hom
    x1_proj_init /= x1_proj_init[2, :]
    
    T_21_34 = np.hstack((R_21, t_21.reshape(3, 1)))
    x2_proj_init = K_c @ T_21_34 @ X1_hom
    x2_proj_init /= x2_proj_init[2, :]
    

    plot2DPoints(image1,image2,x1Data,x2Data,x1_proj_init,x2_proj_init,'Projected (initial)','Initial Residuals')

    # Project the points 3D truth points to each camera  
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1_GT) @ X_w_GT
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_GT) @ X_w_GT
    #x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3_GT) @ X_w_GT
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    #x3_p /= x3_p[2, :]

    plot2DPoints(image1,image2,x1Data,x2Data,x1_p,x2_p,'Projected from 3D point','Before Bundle Adjustment')

    
    # ########################################
    # STEP 3: Bundle Adjustment
    # ########################################
    
    print("\nSTEP 3: Running Bundle Adjustment...")
    print("-" * 70)

    ##para escalar al GT en METROS!!!
    T_c1_w_GT = np.linalg.inv(T_wc1_GT)
    T_c2_w_GT = np.linalg.inv(T_wc2_GT)    
    T12_gt = T_c1_w_GT @ T_wc2_GT
    t12 = T12_gt[0:3, 3]
    normal_t12 = np.linalg.norm(t12)
    print("t12 norm:", np.linalg.norm(t12)) #sanity check 
    #######    
    
    T_21_opt, X1_opt, res_init, res_final = bundleAdjustment(
        x1Data, x2Data, K_c, T_21_init, X1_init, normal_t12
    )

    # ########################################
    # STEP 4: Visualize optimized residuals
    # ########################################
    
    print("\nSTEP 4: Visualizing optimized residuals...")
    print("-" * 70)
    
    X1_hom_opt = np.vstack((X1_opt, np.ones((1, X1_opt.shape[1]))))
    x1_proj_opt = K_c @ np.eye(3, 4) @ X1_hom_opt
    x1_proj_opt /= x1_proj_opt[2, :]
    
    T_21_opt_34 = T_21_opt[:3, :]
    x2_proj_opt = K_c @ T_21_opt_34 @ X1_hom_opt
    x2_proj_opt /= x2_proj_opt[2, :]
    
   
    plot2DPoints(image1,image2,x1Data,x2Data,x1_proj_opt,x2_proj_opt,'Projected (optimized)','Optimized Residuals')

    # ########################################
    # STEP 5: 3D Visualization
    # ########################################
    
    print("\nSTEP 5: 3D Visualization...")
    print("-" * 70)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera 1 (origin)
    T_1 = np.eye(4)
    drawRefSystem(ax, T_1, '-', 'C1')
    
    # Camera 2 (optimized)
    drawRefSystem(ax, T_21_opt, '-', 'C2_opt')
    
    # 3D points (optimized)
    ax.scatter(X1_opt[0, :], X1_opt[1, :], X1_opt[2, :], 
               c='blue', marker='o', label='Optimized points')
    
    # Ground truth for comparison (transform to camera 1 frame)
    T_12_GT = np.linalg.inv(T_wc1_GT) @ T_wc2_GT
    X1_GT = np.linalg.inv(T_wc1_GT) @ X_w_GT
    X1_GT = X1_GT[:3, :] / X1_GT[3, :]
    
    ax.scatter(X1_GT[0, :], X1_GT[1, :], X1_GT[2, :], 
               c='black', marker='^', alpha=0.5, label='Ground truth')
    
    #ax.scatter(X1_init[0, :], X1_init[1, :], X1_init[2, :], 
               #c='red', marker='^', alpha=0.5, label='init points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Reconstruction - Bundle Adjustment Result')
    
    plt.savefig(outPutPath + '02_3d_reconstruction.png', dpi=150, bbox_inches='tight')
    print("* Saved: 02_3d_reconstruction.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("BUNDLE ADJUSTMENT COMPLETE!")
    print("=" * 70)
    input()