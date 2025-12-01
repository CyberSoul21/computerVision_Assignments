#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Methods for laboratory 4, here we have added the function for lab4 to be used as library
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################

import numpy as np
import scipy.linalg as scAlg
import scipy.optimize as scOptim
import matplotlib.pyplot as plt
import cv2

# ############################################################################
# IMPORT YOUR LAB 2 METHODS
# ############################################################################

def fundamentalFromPoses(T_c1_w, T_c2_w, K):
    T_21 = T_c2_w @ np.linalg.inv(T_c1_w)
    R, t = T_21[:3, :3], T_21[:3, 3]
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ t_x @ R @ np.linalg.inv(K)
    return F / F[2, 2]

def essentialMatrix(fundamentalMatrix, K_intrinsict):
    e_Matrix = K_intrinsict.T @ fundamentalMatrix @ K_intrinsict
    U, S, Vt = np.linalg.svd(e_Matrix)
    S = [1,1,0]
    e_Matrix = U @ np.diag(S) @ Vt
    return e_Matrix

def decomposeEssentialMatrix(essentialMatrix): 
    U, _, Vt = np.linalg.svd(essentialMatrix)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    solutions = [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]
    return solutions

def selectCorrectPose(K, x_1, x_2, possible_solutions):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    best_count = 0
    best_pose = None
    best_points = None
    for i, (R, t) in enumerate(possible_solutions):
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangulatePoints(x_1, x_2, P1, P2)
        X_h = np.hstack((X, np.ones((X.shape[0], 1))))
        X_cam2 = (R @ X.T + t.reshape(3, 1)).T
        count = np.sum((X[:, 2] > 0) & (X_cam2[:, 2] > 0))
        if count > best_count:
            best_count = count
            best_pose = (R, t)
            best_points = X
    return best_pose, best_points

def triangulatePoints(x_1, x_2, P1, P2): 
    n = x_1.shape[0]
    X = np.zeros((n, 3))
    for i in range(n):
        u1, v1 = x_1[i]
        u2, v2 = x_2[i]
        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X_h /= X_h[3]
        X[i] = X_h[:3]
    return X


# ############################################################################
# FUNCTIONS FOR ROTATION PARAMETERIZATION Apendix A
# ############################################################################

def crossMatrix(x):
    """
    Create the skew-symmetric (cross-product) matrix from a 3D vector.
    
    -input:
      x: 3D vector [x1, x2, x3]
    -output:
      M: 3x3 skew-symmetric matrix such that M @ v = x × v
    """
    M = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]], dtype=float)
    return M

def crossMatrixInv(M):
    """
    Extract the vector from a skew-symmetric matrix.
    
    -input:
      M: 3x3 skew-symmetric matrix
    -output:
      x: 3D vector [x1, x2, x3]
    """
    x = np.array([M[2, 1], M[0, 2], M[1, 0]])
    return x

def rotationMatrixToTheta(R):
    """
    Convert rotation matrix to so(3) parameterization (3 parameters).
    Uses logarithmic mapping: theta = log(R)
    
    -input:
      R: 3x3 rotation matrix
    -output:
      theta: 3-parameter vector in so(3)
    """
    # IMPORTANT: Cast to float64 to avoid numerical issues
    R = R.astype('float64')
    
    # Logarithmic mapping
    logR = scAlg.logm(R)
    
    # Extract the vector from the skew-symmetric matrix
    theta = crossMatrixInv(logR.real)  # .real to remove tiny imaginary parts
    
    return theta

def thetaToRotationMatrix(theta):
    """
    Convert so(3) parameterization to rotation matrix.
    Uses exponential mapping: R = exp([theta]_x)
    
    -input:
      theta: 3-parameter vector in so(3)
    -output:
      R: 3x3 rotation matrix
    """
    # Create skew-symmetric matrix
    theta_cross = crossMatrix(theta)
    
    # Exponential mapping
    R = scAlg.expm(theta_cross)
    
    return R

def unitVectorFromAngles(theta):
    """
    Get a unit vector from two angles (azimuth and elevation).
    -input:
        theta: (2x1) angles [azimuth; elevation]
    -output:
        u: (3x1) unit vector
    """ 
    azimuth = theta[0]
    elevation = theta[1]
    u = np.array([[np.cos(elevation) * np.cos(azimuth)],
                  [np.cos(elevation) * np.sin(azimuth)],
                  [np.sin(elevation)]])
    return u / np.linalg.norm(u)
# ############################################################################
# BUNDLE ADJUSTMENT RESIDUAL FUNCTION
# ############################################################################

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Residual function for bundle adjustment from two views.
    
    -input:
      Op: Optimization parameters vector
          - Op[0:3]: theta (so(3) parameterization of R_21)
          - Op[3:6]: t (translation of camera 2 w.r.t. camera 1)
          - Op[6:]: X1 (3D points in camera 1 frame, flattened as [X1, Y1, Z1, X2, Y2, Z2, ...])
      x1Data: (3xnPoints) 2D points in image 1 (homogeneous coordinates)
      x2Data: (3xnPoints) 2D points in image 2 (homogeneous coordinates)
      K_c: (3x3) Intrinsic calibration matrix
      nPoints: Number of points
      
    -output:
      res: residuals vector (4*nPoints,)
           For each point: [res_x1, res_y1, res_x2, res_y2]
    """
    
    # ########################################
    # 1. EXTRACT PARAMETERS FROM Op
    # ########################################
    
    # Extract rotation (so(3) parameterization)
    theta = Op[0:3]
    R_21 = thetaToRotationMatrix(theta)
    
    # Extract translation
    t_21 = Op[3:6].reshape(3, 1)
    
    # Extract 3D points (reshape from flat array to 3xN)
    X1 = Op[6:].reshape(3, nPoints)
    
    # ########################################
    # 2. PROJECT 3D POINTS TO BOTH CAMERAS
    # ########################################
    
    # Camera 1: at origin [I | 0]
    X1_homogeneous = np.vstack((X1, np.ones((1, nPoints))))  # 4xN
    x1_projected = K_c @ np.eye(3, 4) @ X1_homogeneous  # 3xN
    
    # Normalize to get pixel coordinates
    x1_projected = x1_projected / x1_projected[2, :]  # Divide by Z
    
    # Camera 2: at pose [R_21 | t_21]
    T_21 = np.hstack((R_21, t_21))  # 3x4
    x2_projected = K_c @ T_21 @ X1_homogeneous  # 3xN
    
    # Normalize to get pixel coordinates
    x2_projected = x2_projected / x2_projected[2, :]
    
    # ########################################
    # 3. COMPUTE RESIDUALS
    # ########################################
    
    # Residuals in image 1 (only x, y components)
    res_x1 = x1Data[0, :] - x1_projected[0, :]  # Nx1
    res_y1 = x1Data[1, :] - x1_projected[1, :]  # Nx1
    
    # Residuals in image 2 (only x, y components)
    res_x2 = x2Data[0, :] - x2_projected[0, :]  # Nx1
    res_y2 = x2Data[1, :] - x2_projected[1, :]  # Nx1
    
    # Stack all residuals into a single vector
    # Order: [res_x1[0], res_y1[0], res_x2[0], res_y2[0], res_x1[1], res_y1[1], ...]
    res = np.empty(4 * nPoints)
    for i in range(nPoints):
        res[4*i]     = res_x1[i]
        res[4*i + 1] = res_y1[i]
        res[4*i + 2] = res_x2[i]
        res[4*i + 3] = res_y2[i]
    
    return res

# ########################################
# BUNDLE ADJUSTMENT OPTIMIZATION
# ########################################

def bundleAdjustment(x1Data, x2Data, K_c, T_21_init, X1_init, normal_t12):
    """
    Perform bundle adjustment optimization.
    
    -input:
      x1Data: (3xN) 2D points in image 1 (homogeneous)
      x2Data: (3xN) 2D points in image 2 (homogeneous)
      K_c: (3x3) Camera calibration matrix
      T_21_init: (4x4) Initial pose of camera 2 w.r.t. camera 1
      X1_init: (3xN) Initial 3D points in camera 1 frame
      
    -output:
      T_21_opt: (4x4) Optimized pose
      X1_opt: (3xN) Optimized 3D points
      res_initial: Initial residuals
      res_final: Final residuals
    """
    
    nPoints = x1Data.shape[1]
    
    # ########################################
    # 1. PREPARE INITIAL PARAMETERS
    # ########################################
    
    # Extract R and t from initial pose
    R_21_init = T_21_init[:3, :3]
    t_21_init = T_21_init[:3, 3]
    
    # Convert rotation to so(3) parameterization
    theta_init = rotationMatrixToTheta(R_21_init)
    
    # Flatten 3D points
    X1_flat = X1_init.flatten()  # [X1, Y1, Z1, X2, Y2, Z2, ...]
    
    # Concatenate all parameters
    Op_init = np.hstack((theta_init, t_21_init, X1_flat))
    
    print(f"Initial parameters: {len(Op_init)} values")
    print(f"  - Rotation (theta): 3 parameters")
    print(f"  - Translation: 3 parameters")
    print(f"  - 3D points: {3 * nPoints} parameters")
    print()
    
    # ########################################
    # 2. COMPUTE INITIAL RESIDUALS
    # ########################################
    
    res_initial = resBundleProjection(Op_init, x1Data, x2Data, K_c, nPoints)
    cost_initial = np.sum(res_initial**2)
    
    print("=== INITIAL STATE ===")
    print(f"Initial cost (sum of squared residuals): {cost_initial:.4f}")
    print(f"RMS reprojection error: {np.sqrt(cost_initial / (4 * nPoints)):.4f} pixels")
    print()
    
    # ########################################
    # 3. RUN OPTIMIZATION
    # ########################################
    
    print("Running bundle adjustment optimization...")
    print("This may take a few seconds...")
    print()
    
    result = scOptim.least_squares(
        resBundleProjection,
        Op_init,
        args=(x1Data, x2Data, K_c, nPoints),
        method='lm',  # Levenberg-Marquardt
        verbose=2      # Show optimization progress
    )
    
    Op_opt = result.x
    res_final = result.fun
    cost_final = result.cost


   ######################################################################
    print()
    print("=== OPTIMIZATION RESULT ===")
    print(f"Final cost: {cost_final:.4f}")
    print(f"RMS reprojection error: {np.sqrt(2 * cost_final / (4 * nPoints)):.4f} pixels")
    print(f"Optimization success: {result.success}")
    print(f"Number of iterations: {result.nfev}")
    print()
    
    # ########################################
    # 4. EXTRACT OPTIMIZED PARAMETERS
    # ########################################
    
    theta_opt = Op_opt[0:3]
    t_21_opt = Op_opt[3:6]
    X1_opt = Op_opt[6:].reshape(3, nPoints)    
    R_21_opt = thetaToRotationMatrix(theta_opt)

    #TODO: Check scale
    ###############################################################
    t_opt = unitVectorFromAngles(t_21_opt).flatten()
    #escalado al GT en METROS!!!
    norm_t_opt = np.linalg.norm(t_opt)
    if norm_t_opt > 1e-8:
        scale2 = normal_t12 / norm_t_opt
    else:
        scale2 = 1.0
        print("Warning: t_opt norm ~0, do not scale.") 
    ###
    # scaled parameters points and translation
    t_21_opt = t_21_opt * scale2
    X1_opt = X1_opt * scale2 
    #Poses scaled
    ################################################################
    
    T_21_opt = np.eye(4)
    T_21_opt[:3, :3] = R_21_opt
    T_21_opt[:3, 3] = t_21_opt
    
    return T_21_opt, X1_opt, res_initial, res_final


    

# ########################################
# N-VIEW BUNDLE ADJUSTMENT - RESIDUAL FUNCTION
# ########################################

def resBundleProjectionNViews(Op, xData_list, K_c, nPoints, nCameras):
    """
    Residual function for bundle adjustment from N views.
    
    CRITICAL DESIGN:
    - Camera 1 is ALWAYS FIXED at origin [I | 0]
    - Cameras 2, 3, ..., N are optimized (N-1 cameras)
    - All cameras see the same 3D points
    
    -input:
      Op: Optimization parameters vector
          Structure: [Camera_2_params, Camera_3_params, ..., Camera_N_params, Points_3D]
          Each camera: [theta[3], t[3]] = 6 parameters
          Points: [X1, Y1, Z1, X2, Y2, Z2, ...] = 3*nPoints parameters
          Total size: 6*(nCameras-1) + 3*nPoints
          
      xData_list: List of (3xnPoints) arrays, one per camera
                  xData_list[0] = x1Data (camera 1 observations)
                  xData_list[1] = x2Data (camera 2 observations)
                  ...
                  
      K_c: (3x3) Camera calibration matrix (assumed same for all cameras)
      nPoints: Number of 3D points
      nCameras: Total number of cameras
      
    -output:
      res: residuals vector (2*nCameras*nPoints,)
           For each point in each camera: [res_x, res_y]
           Order: [cam1_pt1_x, cam1_pt1_y, cam1_pt2_x, cam1_pt2_y, ..., 
                   cam2_pt1_x, cam2_pt1_y, ...]
    """
    
    # ########################################
    # 1. EXTRACT PARAMETERS FROM Op
    # ########################################
    
    # Size check
    expected_size = 6 * (nCameras - 1) + 3 * nPoints
    if len(Op) != expected_size:
        raise ValueError(f"Expected Op size {expected_size}, got {len(Op)}")
    
    # Extract camera poses (cameras 2, 3, ..., N)
    # Camera 1 is fixed at [I | 0]
    camera_poses = []
    
    # Camera 1: Identity (fixed, not in Op)
    T_1 = np.eye(4)
    camera_poses.append(T_1[:3, :])  # [I | 0] as 3x4
    
    # Cameras 2 to N: Extract from Op
    offset = 0
    for cam_idx in range(1, nCameras):  # 1 to N-1 (cameras 2 to N)
        theta = Op[offset:offset+3]
        t = Op[offset+3:offset+6]
        
        R = thetaToRotationMatrix(theta)
        T = np.hstack((R, t.reshape(3, 1)))  # 3x4
        
        camera_poses.append(T)
        offset += 6
    
    # Extract 3D points (in camera 1 frame)
    X1 = Op[offset:].reshape(3, nPoints)
    X1_homogeneous = np.vstack((X1, np.ones((1, nPoints))))  # 4xnPoints
    
    # ########################################
    # 2. PROJECT TO ALL CAMERAS
    # ########################################
    
    projected_points = []
    
    for cam_idx in range(nCameras):
        T_cam = camera_poses[cam_idx]  # 3x4 transformation
        
        # Project: x = K @ T @ X
        x_proj = K_c @ T_cam @ X1_homogeneous  # 3xnPoints
        
        # Normalize (divide by Z)
        x_proj = x_proj / x_proj[2, :]
        
        projected_points.append(x_proj)
    
    # ########################################
    # 3. COMPUTE RESIDUALS
    # ########################################
    
    res = np.empty(2 * nCameras * nPoints)
    
    res_idx = 0
    for cam_idx in range(nCameras):
        xData = xData_list[cam_idx]  # 3xnPoints (observed)
        x_proj = projected_points[cam_idx]  # 3xnPoints (projected)
        
        for pt_idx in range(nPoints):
            res[res_idx]     = xData[0, pt_idx] - x_proj[0, pt_idx]  # residual x
            res[res_idx + 1] = xData[1, pt_idx] - x_proj[1, pt_idx]  # residual y
            res_idx += 2
    
    return res

# ########################################
# N-VIEW BUNDLE ADJUSTMENT - OPTIMIZATION
# ########################################

def bundleAdjustmentNViews(xData_list, K_c, T_init_list, X1_init):
    """
    Perform bundle adjustment for N views.
    
    -input:
      xData_list: List of N arrays (3xM), 2D observations per camera
      K_c: (3x3) Camera calibration matrix
      T_init_list: List of N transformation matrices (4x4)
                   T_init_list[0] = T_1 (should be identity, camera 1 is reference)
                   T_init_list[1] = T_21 (camera 2 w.r.t. camera 1)
                   T_init_list[2] = T_31 (camera 3 w.r.t. camera 1)
                   ...
      X1_init: (3xM) Initial 3D points in camera 1 frame
      
    -output:
      T_opt_list: List of N optimized transformation matrices
      X1_opt: (3xM) Optimized 3D points
      stats: Dictionary with optimization statistics
    """
    
    nCameras = len(xData_list)
    nPoints = xData_list[0].shape[1]
    
    print("\n" + "=" * 80)
    print(f"{'BUNDLE ADJUSTMENT - ' + str(nCameras) + ' VIEWS':^80}")
    print("=" * 80 + "\n")
    
    # ########################################
    # 1. VALIDATE INPUT
    # ########################################
    
    # Check that camera 1 is at identity
    if not np.allclose(T_init_list[0], np.eye(4), atol=1e-6):
        print("WARNING: Camera 1 is not at identity. Forcing T_1 = I.")
        T_init_list[0] = np.eye(4)
    
    # Check all data has same number of points
    for i, xData in enumerate(xData_list):
        if xData.shape[1] != nPoints:
            raise ValueError(f"Camera {i+1} has {xData.shape[1]} points, expected {nPoints}")
    
    print(f"Configuration:")
    print(f"  Number of cameras:      {nCameras}")
    print(f"  Number of points:       {nPoints}")
    print(f"  Fixed cameras:          1 (Camera 1 at origin)")
    print(f"  Optimized cameras:      {nCameras - 1}")
    print()
    
    # ########################################
    # 2. PREPARE INITIAL PARAMETERS
    # ########################################
    
    print("Preparing optimization parameters...")
    print("-" * 80)
    
    # Build parameter vector
    Op_init = []
    
    # Add camera parameters (cameras 2 to N)
    for cam_idx in range(1, nCameras):
        R = T_init_list[cam_idx][:3, :3]
        t = T_init_list[cam_idx][:3, 3]
        
        theta = rotationMatrixToTheta(R)
        
        Op_init.extend(theta)  # 3 params
        Op_init.extend(t)      # 3 params
    
    # Add 3D points
    Op_init.extend(X1_init.flatten())
    
    Op_init = np.array(Op_init)
    
    n_camera_params = 6 * (nCameras - 1)
    n_point_params = 3 * nPoints
    n_total_params = n_camera_params + n_point_params
    
    print(f"Parameter breakdown:")
    print(f"  Camera parameters:      {n_camera_params} ({nCameras-1} cameras × 6 DOF)")
    print(f"  Point parameters:       {n_point_params} ({nPoints} points × 3 coords)")
    print(f"  Total parameters:       {n_total_params}")
    print()
    
    n_residuals = 2 * nCameras * nPoints
    print(f"Residuals:")
    print(f"  Per point per camera:   2 (x, y)")
    print(f"  Total residuals:        {n_residuals} ({nCameras} cameras × {nPoints} points × 2)")
    print()
    
    # ########################################
    # 3. COMPUTE INITIAL COST
    # ########################################
    
    res_initial = resBundleProjectionNViews(Op_init, xData_list, K_c, nPoints, nCameras)
    cost_initial = np.sum(res_initial**2)
    rms_initial = np.sqrt(cost_initial / n_residuals)
    
    print("Initial state:")
    print(f"  Total cost:             {cost_initial:.4f}")
    print(f"  RMS reprojection error: {rms_initial:.4f} pixels")
    print()
    
    # ########################################
    # 4. RUN OPTIMIZATION
    # ########################################
    
    print("Running bundle adjustment optimization...")
    print("This may take 10-60 seconds depending on number of cameras and points...")
    print("-" * 80)
    
    result = scOptim.least_squares(
        resBundleProjectionNViews,
        Op_init,
        args=(xData_list, K_c, nPoints, nCameras),
        method='lm',      # Levenberg-Marquardt
        verbose=2,        # Show progress
        max_nfev=300,     # Maximum iterations
        ftol=1e-8,        # Function tolerance
        xtol=1e-8         # Parameter tolerance
    )
    
    Op_opt = result.x
    cost_final = result.cost
    rms_final = np.sqrt(2 * cost_final / n_residuals)
    
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"Final cost:             {cost_final:.6f}")
    print(f"RMS reprojection error: {rms_final:.6f} pixels")
    print(f"Success:                {result.success}")
    print(f"Iterations:             {result.nfev}")
    print(f"Termination reason:     {result.message}")
    print()
    
    improvement = 100 * (1 - cost_final / cost_initial)
    print(f"Cost improvement:       {cost_initial:.4f} → {cost_final:.6f} ({improvement:.2f}%)")
    print(f"RMS improvement:        {rms_initial:.4f} → {rms_final:.6f} pixels")
    print()
    
    # ########################################
    # 5. EXTRACT OPTIMIZED PARAMETERS
    # ########################################
    
    T_opt_list = []
    
    # Camera 1: Identity (unchanged)
    T_opt_list.append(np.eye(4))

    scale2 = 1.0 #Added
    # Cameras 2 to N
    offset = 0
    for cam_idx in range(1, nCameras):
        theta = Op_opt[offset:offset+3]
        t = Op_opt[offset+3:offset+6]
        
        R = thetaToRotationMatrix(theta)
       
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        T_opt_list.append(T)
        offset += 6
    
    # 3D points
    X1_opt = Op_opt[offset:].reshape(3, nPoints)
    X1_opt = X1_opt*scale2 #Added

    # Statistics
    stats = {
        'nCameras': nCameras,
        'nPoints': nPoints,
        'cost_initial': cost_initial,
        'cost_final': cost_final,
        'rms_initial': rms_initial,
        'rms_final': rms_final,
        'success': result.success,
        'iterations': result.nfev,
        'improvement_pct': improvement
    }
    
    return T_opt_list, X1_opt, stats

# ########################################
# SCALE FIXING (for arbitrary scale reconstruction)
# ########################################

def fixScaleNViews(T_opt_list, X1_opt, T_GT_list):
    """
    Fix scale of reconstruction using ground truth.
    Uses the distance between cameras 1 and 2.
    
    -input:
      T_opt_list: List of optimized transformations (arbitrary scale)
      X1_opt: Optimized 3D points (arbitrary scale)
      T_GT_list: List of ground truth transformations
      
    -output:
      T_scaled_list: Scaled transformations
      X1_scaled: Scaled 3D points
      scale_factor: The computed scale factor
    """
    
    print("\n" + "=" * 80)
    print(f"{'SCALE FIXING':^80}")
    print("=" * 80 + "\n")
    
    # Compute scale from camera 1 to camera 2 baseline
    t_21_estimated = T_opt_list[1][:3, 3]
    t_21_GT = T_GT_list[1][:3, 3]
    
    scale_GT = np.linalg.norm(t_21_GT)
    scale_estimated = np.linalg.norm(t_21_estimated)
    
    if scale_estimated < 1e-6:
        print("ERROR: Estimated baseline is too small, cannot compute scale!")
        scale_factor = 1.0
    else:
        scale_factor = scale_GT / scale_estimated
    
    print(f"Using baseline between Camera 1 and Camera 2:")
    print(f"  Ground truth ||t_21||:  {scale_GT:.6f}")
    print(f"  Estimated ||t_21||:     {scale_estimated:.6f}")
    print(f"  Scale factor:           {scale_factor:.6f}")
    print()
    
    # Apply scale to all transformations
    T_scaled_list = []
    T_scaled_list.append(T_opt_list[0])  # Camera 1 unchanged (identity)
    
    for i in range(1, len(T_opt_list)):
        T_scaled = T_opt_list[i].copy()
        T_scaled[:3, 3] *= scale_factor
        T_scaled_list.append(T_scaled)
    
    # Apply scale to 3D points
    X1_scaled = X1_opt * scale_factor
    
    print(f"✓ Scale applied to {len(T_opt_list)-1} cameras and {X1_opt.shape[1]} points")
    print()
    
    return T_scaled_list, X1_scaled, scale_factor


# ########################################
# MAIN FUNCTION
# ########################################   
if __name__ == '__main__':
    print("=" * 50)
    print()
    print("This module contains the residual function and optimization")
    print("for bundle adjustment from two camera views.")
    print()
