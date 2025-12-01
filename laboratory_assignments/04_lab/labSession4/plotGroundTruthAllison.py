#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Bundle Adjustment and Multiview Geometry
#
# Date: 26 October 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio


######## Reuse from labSession2 ########

def computeFfromMatches(x1, x2):
    nMatches = x1.shape[1]
    if x2.shape[1] != nMatches:
        raise ValueError("The number of points in both images must be the same.")
    if nMatches < 8:
        raise ValueError("At least 8 matches are required to compute the Fundamental Matrix.")
    
    A = np.zeros((nMatches, 9))
    for i in range(nMatches):
        X1 = x1[0, i]
        Y1 = x1[1, i]
        X2 = x2[0, i]
        Y2 = x2[1, i]
        A[i, :] = [X2 * X1, X2 * Y1, X2,
                   Y2 * X1, Y2 * Y1, Y2,
                   X1,      Y1,      1]
    
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1, :].reshape(3, 3) 

    # Enforce rank-2 constraint
    U_F, S_F, Vh_F = np.linalg.svd(F)
    S_F[2] = 0
    F_rank2 = U_F @ np.diag(S_F) @ Vh_F

    return F_rank2

def computeEfromF(F, K_c1, K_c2):
    E = K_c2.T @ F @ K_c1
    return E

def nMatchesSeenTwice(R, t, x1, x2, K_c1, K_c2):
    P_1 = K_c1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_2 = K_c2 @ np.hstack((R, t.reshape(3, 1)))

    points3d = triang_points(x1, x2, P_1, P_2)
    numPositiveDepth = 0
    for i in range(points3d.shape[1]):  
        point3d = points3d[:, i]
        z1 = point3d[2]
        point3d_cam2 = R @ point3d[0:3] + t
        z2 = point3d_cam2[2]
        if z1 > 0 and z2 > 0:
            numPositiveDepth += 1
    return numPositiveDepth

def getTFromE(E):
    U, S, Vh = np.linalg.svd(E)
    t = U[:, 2]
    if np.linalg.norm(t) > 0:
        t = t / np.linalg.norm(t)

    W = np.array([[0, -1, 0],
                  [1,  0, 0], 
                  [0,  0, 1]])
    
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh    
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    bestR = None
    bestnSeenTwice = -1
    minusT = False
    for R in [R1, R2]:
        n = nMatchesSeenTwice(R, t, x1Data, x2Data, K_c, K_c)
        n_neg = nMatchesSeenTwice(R, -t, x1Data, x2Data, K_c, K_c)
        if n > bestnSeenTwice or n_neg > bestnSeenTwice:
            if n_neg > n:
                bestnSeenTwice = n_neg
                bestR = R
                minusT = True
            else:
                bestnSeenTwice = n
                bestR = R
                minusT = False

    R = bestR
    if minusT:
        t = -t

    print("Best T found, nSeenTwice = ", bestnSeenTwice)
    return ensamble_T(R, t)


def getFmatrixFromT(T21, K_c1, K_c2):
    R = T21[0:3, 0:3]
    t = T21[0:3, 3].ravel()
    t_skew = np.array([[    0, -t[2],  t[1]],
                       [ t[2],     0, -t[0]],
                       [-t[1],  t[0],     0]])
    E = t_skew @ R
    return np.linalg.inv(K_c2).T @ E @ np.linalg.inv(K_c1)

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def buildRowsA(x, P):
    A = np.zeros((2,4))
    for i in range(4):
        A[0,i] = P[2,i] * x[0] - P[0,i]
        A[1,i] = P[2,i] * x[1] - P[1,i]
    return A

def triang_points(x1,x2,P1,P2):
    # matrix P 3x3
    # x1 y x2 2x num puntos
    points3d = []
    nMatches = x1.shape[1]
    for i in range(nMatches):
        A = np.zeros((4,4))
        A[:2,:] = buildRowsA(x1[:,i],P1)
        A[2:,:] = buildRowsA(x2[:,i],P2)

        u, s, vh = np.linalg.svd(A) # svd function returns vh which is the tranpose version of V matrix.
        l_ls = vh[-1, :]
        l_ls = l_ls/l_ls[3]

        points3d.append(l_ls)
        # print("Point found: ", l_ls)

    return np.array(points3d).transpose()

############ fin labSession2 functions ##############

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return np.array(x, dtype=np.float64)

def crossMatrix(x):
    M = np.array([[0,
    -x[2], x[1]],
    [x[2], 0,
    -x[0]],
    [-x[1], x[0], 0]], dtype=np.float64)
    return M

def R_from_theta(theta):
    return scAlg.expm(crossMatrix(theta))

def theta_from_R(R):
    theta = crossMatrixInv(scAlg.logm(R))
    return theta

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

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
    Op: Optimization parameters: this must include a
    paramtrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    x2Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    K_c: (3x3) Intrinsic calibration matrix
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)
    """
    # Extract the parameters to optimize

    rVec = Op[0:3] # theta
    # extract traslation as unit vector from two angles
    tVec = Op[3:5]
    t = unitVectorFromAngles(tVec).flatten()
    #extract 3D points
    X = Op[5:].reshape((3, nPoints))
    # Convert rotation vector to rotation matrix
    R = R_from_theta(rVec)

    #projection matrices
    P1 = K_c @ np.hstack([np.eye(3), np.zeros((3,1))])        # (3x4)
    T2 = ensamble_T(R, t)
    P2 = K_c @ T2[0:3, :]                           # (3x4)
    # Homogeneous coordinates of 3D points
    Xh = np.vstack((X, np.ones((1, nPoints))))            # (4xN)
    x1_proj = P1 @ Xh                                # (3xN)
    x2_proj= P2 @ Xh                                # (3xN)                                    
    

    # Compute the residuals
    res_x1 = x1Data[0:2, :] - (x1_proj[0:2, :] / x1_proj[2, :])
    res_x2 = x2Data[0:2, :] - (x2_proj[0:2, :] / x2_proj[2, :])

    res_x1_total = np.sum(np.abs(res_x1))
    res_x2_total = np.sum(np.abs(res_x2))

    print("\n residuals 1:")
    print(res_x1_total)
    print("\n residuals 2:")
    print(res_x2_total)

    res = np.hstack((res_x1.flatten(), res_x2.flatten()))

    return res

def resBundleProjection3Views12DoF(Op, x1Data, x2Data, x3Data, K_c, nPoints):
    """
    Residual function for bundle adjustment with 3 views (C1, C2, C3),
    with full 12 DoF for the two relative poses:

        - theta_21 (3): rotation from camera 1 to camera 2
        - t_21     (3): translation from camera 1 to camera 2
        - theta_31 (3): rotation from camera 1 to camera 3
        - t_31     (3): translation from camera 1 to camera 3
        - X_c1     (3*N): 3D points in camera-1 reference frame

    Op = [ theta_21(3), t_21(3), theta_31(3), t_31(3), X(3*N) ]
    """

    # --- Extraer parámetros ---
    theta_21 = Op[0:3]           # rotación C1 -> C2
    t_21     = Op[3:6]           # traslación C1 -> C2
    theta_31 = Op[6:9]           # rotación C1 -> C3
    t_31     = Op[9:12]          # traslación C1 -> C3
    X_c1     = Op[12:].reshape((3, nPoints))  # puntos 3D en ref. C1

    # R
    R_21 = R_from_theta(theta_21)   
    R_31 = R_from_theta(theta_31)

    # T
    T_21 = ensamble_T(R_21, t_21)   # cam2 from cam1
    T_31 = ensamble_T(R_31, t_31)   # cam3 from cam1

    # --- Matrices de proyección ---
    # Cámara 1: referencia fija
    P1 = K_c @ np.hstack([np.eye(3), np.zeros((3,1))])

    
    P2 = K_c @ T_21[0:3, :]   # (3x4)
    P3 = K_c @ T_31[0:3, :]   # (3x4)

  
    Xh = np.vstack((X_c1, np.ones((1, nPoints))))   # (4xN)

    # `projection of 3D points`
    x1_proj = P1 @ Xh
    x2_proj = P2 @ Xh
    x3_proj = P3 @ Xh

   
    x1_proj /= x1_proj[2, :]
    x2_proj /= x2_proj[2, :]
    x3_proj /= x3_proj[2, :]

    # Residuals 2D
    res_x1 = x1Data[0:2, :] - x1_proj[0:2, :]
    res_x2 = x2Data[0:2, :] - x2_proj[0:2, :]
    res_x3 = x3Data[0:2, :] - x3_proj[0:2, :]

   
    res_x1_total = np.sum(np.abs(res_x1))
    res_x2_total = np.sum(np.abs(res_x2))
    res_x3_total = np.sum(np.abs(res_x3))

    print("\nResiduals view 1 (sum abs):", res_x1_total)
    print("Residuals view 2 (sum abs):", res_x2_total)
    print("Residuals view 3 (sum abs):", res_x3_total)


    residuals = np.hstack((res_x1.flatten(), res_x2.flatten(), res_x3.flatten()))
    return residuals



def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def compute_reproj_and_plot(image, xObs, xProj, title=None, show_rmse=True, figsize=(7,6)):
    """
    Dibuja en la imagen las proyecciones y residuals.
    - image: array leído por cv2 (BGR) o similar. Convertimos a RGB para plt.
    - xObs: 3xN puntos observados (homog)
    - xProj: 3xN puntos proyectados (homog)
    Devuelve: dict con per_point_errors y rmse.
    """
    # Asegurar shapes
    if xObs.shape[0] == 2:
        xObs = np.vstack((xObs, np.ones((1, xObs.shape[1]))))
    if xProj.shape[0] == 2:
        xProj = np.vstack((xProj, np.ones((1, xProj.shape[1]))))

    # per-point euclidiano en píxeles
    diffs = xObs[0:2, :] - xProj[0:2, :]
    per_point = np.linalg.norm(diffs, axis=0)
    rmse = float(np.sqrt(np.mean(per_point**2)))

    # Plot
    plt.figure(figsize=figsize)
    # si es imagen BGR (cv2) convertir a RGB
    try:
        img_show = image.copy()
        if img_show.shape[2] == 3:
            img_show = img_show[:,:,::-1]  # BGR -> RGB
    except Exception:
        img_show = image
    plt.imshow(img_show, cmap='gray', vmin=0, vmax=255)
    # residual lines (negro)
    for k in range(xObs.shape[1]):
        plt.plot([xObs[0,k], xProj[0,k]], [xObs[1,k], xProj[1,k]], 'k-', linewidth=0.6)
    # puntos proyectados en azul, observados en rojo
    plt.plot(xProj[0,:], xProj[1,:], 'bo', markersize=4, label='projected')
    plt.plot(xObs[0,:], xObs[1,:], 'rx', markersize=6, label='observed')
    plt.legend(loc='upper right')

    # RMSE text
    if show_rmse:
        plt.text(0.02, 0.98, f'RMSE = {rmse:.3f} px', transform=plt.gca().transAxes,
                 fontsize=12, color='yellow', backgroundcolor='black', va='top')

    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

    return {'per_point': per_point, 'rmse': rmse}

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    basePath = "labSession4/"
    # Load ground truth
    T_wc1 = np.loadtxt(basePath + 'T_w_c1.txt')
    T_wc2 = np.loadtxt(basePath + 'T_w_c2.txt')
    T_wc3 = np.loadtxt(basePath + 'T_w_c3.txt')
    K_c = np.loadtxt(basePath + 'K_c.txt')
    X_w = np.loadtxt(basePath + 'X_w.txt')

    #3D points reconstructed from the three views // Matches
    x1Data = np.loadtxt(basePath + 'x1Data.txt')
    x2Data = np.loadtxt(basePath + 'x2Data.txt')
    x3Data = np.loadtxt(basePath + 'x3Data.txt')

##### Chose F matrix (given or computed) #####
    givenF = False
    if givenF:
        F_21 = np.loadtxt('F_21.txt')
    else:
        F_21 = computeFfromMatches(x1Data, x2Data)

    E_21 = computeEfromF(F_21, K_c, K_c)

    # Relative pose T_21 (camera 2 from cam 1)then R_21, t_21
    T_21 = getTFromE(E_21)
    R_21 = T_21[0:3, 0:3]
    t_21 = T_21[0:3, 3]
    theta0 = theta_from_R(R_21)          # θ inicial desde R
    t0 = t_21.copy()                 # t inicial (escala arbitraria)
    # Initial triangulation of X1 with P1=K[I|0], P2=K[R|t]
    P1 = K_c @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K_c @ np.hstack([R_21, t0.reshape(3,1)])
    X1_h = triang_points(x1Data, x2Data, P1, P2)    # (4xN)
    X1_0 = (X1_h[0:3, :] / X1_h[3:4, :]) # initial points normalized (3xN)

    # test bundle adjustment residuals with GT
    nPoints = X1_0.shape[1]
    t0_unitario = t0 / np.linalg.norm(t0)
    # vector t as two angles
    az = np.arctan2(t0_unitario[1], t0_unitario[0])
    elev = np.arcsin(t0_unitario[2])
    t0_angles = np.array([az, elev])
    #vector initial parameters for ooptimization [rVec(3), tVec(2), X1(3xnPoints)
    Op0 = np.hstack((theta0, t0_angles, X1_0.flatten()))
    res0 = resBundleProjection(Op0, x1Data, x2Data, K_c, nPoints)
    print("Initial residuals with GT parameters:", np.linalg.norm(res0))

    
    ##para escalar al GT en METROS!!!
    T_c1_w_GT = np.linalg.inv(T_wc1)
    T_c2_w_GT = np.linalg.inv(T_wc2)    
    T12_gt = T_c1_w_GT @ T_wc2
    t12 = T12_gt[0:3, 3]
    normal_t12 = np.linalg.norm(t12)
    print("t12 norm:", np.linalg.norm(t12)) #sanity check 
        
    t21 = T_21[0:3,3]
    t21_escalado= t21*normal_t12
    T_21_escalado = ensamble_T(T_21[0:3, 0:3], t21_escalado)
    T_c2_w_escalado = T_21_escalado @ T_c1_w_GT
    T_w_c2_escalado = np.linalg.inv(T_c2_w_escalado)
    P_2_escalado = K_c @ np.eye(3,4) @ T_c2_w_escalado
    #######

    # #Plot the 3D cameras and the 3D points
    # fig3D = plt.figure(1)

    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    # drawRefSystem(ax, T_wc1, '-', 'C1')
    # drawRefSystem(ax, T_wc2, '-', 'C2')
    # drawRefSystem(ax, T_wc3, '-', 'C3')

    # #plotNumbered3DPoints(ax, X_w, 'r', 0.1)
    # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.', alpha=0.4)

    # #Matplotlib does not correctly manage the axis('equal')
    # xFakeBoundingBox = np.linspace(0, 4, 2)
    # yFakeBoundingBox = np.linspace(0, 4, 2)
    # zFakeBoundingBox = np.linspace(0, 4, 2)
    # plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    # print('Close the figure to continue. Left button for orbit, right button for zoom.')
    # plt.show()


    #Read the images
    path_image_1 = basePath + 'image1.png'
    path_image_2 = basePath + 'image2.png'
    path_image_3 = basePath + 'image3.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)


    # Construct the matches
    kpCv1 = []
    kpCv2 = []
    kpCv3 = []
    for kPoint in range(x1Data.shape[1]):
        kpCv1.append(cv2.KeyPoint(x1Data[0, kPoint], x1Data[1, kPoint],1))
        kpCv2.append(cv2.KeyPoint(x2Data[0, kPoint], x2Data[1, kPoint],1))
        kpCv3.append(cv2.KeyPoint(x3Data[0, kPoint], x3Data[1, kPoint],1))

    matchesList12 = np.hstack((np.reshape(np.arange(0, x1Data.shape[1]),(x2Data.shape[1],1)),
                                        np.reshape(np.arange(0, x1Data.shape[1]), (x1Data.shape[1], 1)),np.ones((x1Data.shape[1],1))))

    matchesList13 = matchesList12
    dMatchesList12 = indexMatrixToMatchesList(matchesList12)
    dMatchesList13 = indexMatrixToMatchesList(matchesList13)

    imgMatched12 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_2, kpCv2, dMatchesList12,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    imgMatched13 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_3, kpCv3, dMatchesList13,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # plt.figure(2)
    # plt.imshow(imgMatched12)
    # plt.title("{} matches between views 1 and 2".format(len(dMatchesList12)))
    # plt.draw()

    # plt.figure(3)
    # plt.imshow(imgMatched13)
    # plt.title("{} matches between views 1 and 3".format(len(dMatchesList13)))
    # print('Close the figures to continue.')
    # plt.show()

    # Project the points
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ X_w
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_w
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]

    # Projecttion of the 3D initial points X1_0 with initial pose T_21 -> E
    X1_0_h = np.vstack((X1_0, np.ones((1, X1_0.shape[1]))))
    P1_init = K_c @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2_init = K_c @ np.hstack([R_21, t0_unitario.reshape(3,1)])
    x1_p_init = P1_init @ X1_0_h
    x2_p_init = P2_init @ X1_0_h
    x1_p_init /= x1_p_init[2, :]
    x2_p_init /= x2_p_init[2, :]
    
     # --- Visualize before BA ---
    compute_reproj_and_plot(image_pers_1, x1Data, x1_p_init, title='Image 1 - BEFORE BA')
    compute_reproj_and_plot(image_pers_2, x2Data, x2_p_init, title='Image 2 - BEFORE BA')
    compute_reproj_and_plot(image_pers_3, x3Data, x3_p, title='Image 3 - BEFORE BA')


    # Plot the 2D points
    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 1')
    plt.draw()

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.title('Image 2')
    plt.draw()

    plt.figure(6)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.title('Image 3')
    print('Close the figures to continue.')
    plt.show()

    # Plot the projections with initial parameters using E, no GT
    plt.figure(7)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p_init, 'k-')
    plt.plot(x1_p_init[0, :], x1_p_init[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plt.title('Image 1 - Initial projection from E')
    plt.draw()

    plt.figure(8)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p_init, 'k-')
    plt.plot(x2_p_init[0, :], x2_p_init[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plt.title('Image 2 - Initial projection from E')
    plt.draw()
    print('Close the figures to continue.')

    # Optimize with bundle adjustment
    print("Starting bundle adjustment...")
    resultBA = scOptim.least_squares(
        resBundleProjection, 
        Op0,
        args=(x1Data, x2Data, K_c, nPoints), method = 'lm') #Levenberg-Marquardt pero podria ser otro metodo
    
    optimize = resultBA.x # vector of optimized parameters thats includes rVec, tVec, X1
    
    #extract optimized parameters
    rVec_opt = optimize[0:3]
    tVec_opt = optimize[3:5]
    X_opt = optimize[5:].reshape((3, nPoints))
    R_opt = R_from_theta(rVec_opt)
    t_opt = unitVectorFromAngles(tVec_opt).flatten()

    #escalado al GT en METROS!!!
    norm_t_opt = np.linalg.norm(t_opt)
    if norm_t_opt > 1e-8:
        scale2 = normal_t12 / norm_t_opt
    else:
        scale2 = 1.0
        print("Warning: t_opt norm ~0, do not scale.") 
    # scaled parameters points and translation
    t_opt = t_opt * scale2
    X_opt = X_opt * scale2 
    #Poses scaled
    T_21_opt = ensamble_T(R_opt, t_opt)
    T_c1_w_GT = np.linalg.inv(T_wc1) # tb vale para C3
    T_c2_w_opt = T_21_opt @ T_c1_w_GT
    T_w_c2_opt = np.linalg.inv(T_c2_w_opt)

    #Projection with optimized SCALED parameters
    P1_opt = K_c @ np.hstack([np.eye(3), np.zeros((3,1))]) #tb vale sin escala, no varía
    P2_opt_scaled = K_c @ np.hstack([R_opt, t_opt.reshape(3,1)])
    Xh_opt_scaled = np.vstack((X_opt, np.ones((1, nPoints))))
    x1_opt_scaled = P1_opt @ Xh_opt_scaled
    x2_opt_scaled = P2_opt_scaled @ Xh_opt_scaled
    x1_opt_scaled /= x1_opt_scaled[2, :]
    x2_opt_scaled /= x2_opt_scaled[2, :]


    #projection with optimized parameters
    X_opt_h = np.vstack((X_opt, np.ones((1, nPoints))))
 
    P2_opt = K_c @ np.hstack([R_opt, t_opt.reshape(3,1)])
    x1_p_opt = P1_opt @ X_opt_h
    x2_p_opt = P2_opt @ X_opt_h
    x1_p_opt /= x1_p_opt[2, :]
    x2_p_opt /= x2_p_opt[2, :]  
    compute_reproj_and_plot(image_pers_1, x1Data, x1_p_opt, title='Image 1 - AFTER BA (2-view)')
    compute_reproj_and_plot(image_pers_2, x2Data, x2_p_opt, title='Image 2 - AFTER BA (2-view)')

    #visualize optimized projections
    plt.figure(9)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p_opt, 'k-')
    plt.plot(x1_p_opt[0, :], x1_p_opt[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plt.title('Image 1 - Optimized projection')     
    plt.draw()

    plt.figure(10)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p_opt, 'k-')                
    plt.plot(x2_p_opt[0, :], x2_p_opt[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plt.title('Image 2 - Optimized projection')
    print('Close the figures to finish.')   

    ##### BA with 3 views and 12 DoF #####

   
    theta_21_0 = theta0          # theta_from_R(R_21)
    t_21_0     = t0.copy()       #  t inicial de E (sin escala GT)

    # C3 desde C1 (GT)
    T_c3_w_GT = np.linalg.inv(T_wc3)
    T_31_GT   = T_c3_w_GT @ T_wc1   # cam3 desde cam1 (GT)

    R_31_0 = T_31_GT[0:3, 0:3]
    t_31_0 = T_31_GT[0:3, 3]

    theta_31_0 = theta_from_R(R_31_0)

    # Using X optimized from 2-view BA as initial 3D points
    X0_3views = X_opt   # (3xN)

    # Op = [ theta_21(3), t_21(3), theta_31(3), t_31(3), X(3*N) ]
    Op0_12 = np.hstack((
        theta_21_0,
        t_21_0,
        theta_31_0,
        t_31_0,
        X0_3views.flatten()
    ))

    print("Starting 3-view BA (12 DoF)...")
    resultBA3 = scOptim.least_squares(
        resBundleProjection3Views12DoF,
        Op0_12,
        args=(x1Data, x2Data, x3Data, K_c, nPoints),
        method='lm'
    )

    Op_12_opt = resultBA3.x

    # extrtact optimized parameters
    theta_21_opt = Op_12_opt[0:3]
    t_21_opt     = Op_12_opt[3:6]
    theta_31_opt = Op_12_opt[6:9]
    t_31_opt     = Op_12_opt[9:12]
    X_opt_3views = Op_12_opt[12:].reshape((3, nPoints))

    R_21_opt = R_from_theta(theta_21_opt)
    R_31_opt = R_from_theta(theta_31_opt)

    # ====================== scaled ======================

    # Normalized C1->C2 after  BA 3 
    norm_t21_opt = np.linalg.norm(t_21_opt)

    # Scale factor to match GT t12 norm
    if norm_t21_opt > 1e-8:
        scaleBA3 = normal_t12 / norm_t21_opt
    else:
        scaleBA3 = 1.0
        print("Warning: t_21_opt norm ~0, do not scale.")

    # scaled parameters
    t_21_opt_s   = t_21_opt   * scaleBA3
    t_31_opt_s   = t_31_opt   * scaleBA3
    X_opt_3_s    = X_opt_3views * scaleBA3

    # Scaled camera poses
    T_21_opt_s = ensamble_T(R_21_opt, t_21_opt_s)
    T_31_opt_s = ensamble_T(R_31_opt, t_31_opt_s)

    print("Optimized T_21 (3-view BA, scaled):")
    print(T_21_opt_s)
    print("Optimized T_31 (3-view BA, scaled):")
    print(T_31_opt_s)

    # ====================== POSES in GT ======================

    # C2 BA3 en mundo
    T_c2_w_BA3  = T_21_opt_s @ T_c1_w_GT
    T_wc2_BA3   = np.linalg.inv(T_c2_w_BA3)

    # C3 BA3 en mundo
    T_c3_w_BA3  = T_31_opt_s @ T_c1_w_GT
    T_wc3_BA3   = np.linalg.inv(T_c3_w_BA3)

    # Puntos 3D BA3 en mundo
    X_opt_3_h_w = T_wc1 @ np.vstack((X_opt_3_s, np.ones((1, nPoints))))
    X_opt_3_w   = X_opt_3_h_w[0:3, :] / X_opt_3_h_w[3, :]

    # ====================== 3D BA 12 Dof ======================

    fig3D_BA3 = plt.figure(12)
    ax3 = plt.axes(projection='3d', adjustable='box')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    drawRefSystem(ax3, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax3, T_wc1, '-', 'C1_GT')
    drawRefSystem(ax3, T_wc2, '-', 'C2_GT')
    drawRefSystem(ax3, T_wc3, '-', 'C3_GT')

    # Cámaras estimadas con BA 3 vistas
    drawRefSystem(ax3, T_wc2_BA3, '--', 'C2_BA3')
    drawRefSystem(ax3, T_wc3_BA3, '--', 'C3_BA3')

    # Puntos: GT y BA3
    ax3.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.', alpha=0.4, label='GT points')
    ax3.scatter(X_opt_3_w[0, :], X_opt_3_w[1, :], X_opt_3_w[2, :], marker='.', alpha=0.4, label='BA3 points')

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    ax3.legend()
    plt.title('3-view BA (scaled with GT) vs Ground Truth')
    plt.show()

        # ====================== PROYECCIONES CON LA SOLUCIÓN BA 3 VISTAS ======================

    # Construir T_21 y T_31 con la solución del BA3 (sin escalar o ya escaladas, como prefieras)
    T_21_opt = ensamble_T(R_21_opt, t_21_opt)
    T_31_opt = ensamble_T(R_31_opt, t_31_opt)

    # Matrices de proyección de las 3 cámaras
    P1_BA3 = K_c @ np.hstack([np.eye(3), np.zeros((3,1))])   # C1 fija
    P2_BA3 = K_c @ T_21_opt[0:3, :]                          # C2 desde C1
    P3_BA3 = K_c @ T_31_opt[0:3, :]                          # C3 desde C1

    # Puntos 3D optimizados en homogéneas
    Xh_BA3 = np.vstack((X_opt_3views, np.ones((1, nPoints))))  # (4xN)

    # Proyección en cada cámara
    x1_BA3 = P1_BA3 @ Xh_BA3
    x2_BA3 = P2_BA3 @ Xh_BA3
    x3_BA3 = P3_BA3 @ Xh_BA3

    # Normalizar
    x1_BA3 /= x1_BA3[2, :]
    x2_BA3 /= x2_BA3[2, :]
    x3_BA3 /= x3_BA3[2, :]
    compute_reproj_and_plot(image_pers_1, x1Data, x1_BA3, title='Image 1 - AFTER BA (3-view)')
    compute_reproj_and_plot(image_pers_2, x2Data, x2_BA3, title='Image 2 - AFTER BA (3-view)')
    compute_reproj_and_plot(image_pers_3, x3Data, x3_BA3, title='Image 3 - AFTER BA (3-view)')


    # ====================== Residuals======================

    # Imagen 1
    plt.figure(20)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_BA3, 'k-')              # líneas negras: residual
    plt.plot(x1_BA3[0, :], x1_BA3[1, :], 'bo')      # proyección BA3 (azul)
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')      # puntos medidos (rojo)
    plt.title('Image 1 - Residuals after 3-view BA')
    plt.draw()

    # Imagen 2
    plt.figure(21)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_BA3, 'k-')
    plt.plot(x2_BA3[0, :], x2_BA3[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plt.title('Image 2 - Residuals after 3-view BA')
    plt.draw()

    # Imagen 3
    plt.figure(22)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_BA3, 'k-')
    plt.plot(x3_BA3[0, :], x3_BA3[1, :], 'bo')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plt.title('Image 3 - Residuals after 3-view BA')
    print('Close the figures to continue.')
    plt.show()



   ##########Perspective-N-Point pose estimation of camera three ##########
    # PnP of the caemera 3 with the optimized 3D respect to camera 1
    objectPoints = X_opt.T.astype(np.float64)
    imagePoints = np.ascontiguousarray(x3Data[0:2, :].T).reshape((x3Data.shape[1],1,2))
    distCoeffs = np.zeros((4,1)) # No distortion
    retval, rvec_c3_c1, tvec_c3_c1 = cv2.solvePnP(objectPoints, imagePoints, K_c, distCoeffs,flags=cv2.SOLVEPNP_EPNP)
    rvec_c3_c1 = rvec_c3_c1.reshape(3, 1)
    R31, _ = cv2.Rodrigues(rvec_c3_c1)         # (3x3)
    R31 = np.asarray(R31, dtype=np.float64)
    t31 = tvec_c3_c1.reshape(3)    
    T_31 = ensamble_T(R31, t31) # camera pose 3 from camera 1 this is need fot compare with GT
    print("Estimated pose of camera 3 from PnP with optimized points from BA:")
    print(T_31)

    #Compare with GT
    T_c1_w_GT = np.linalg.inv(T_wc1)
    T_c3_w_GT = np.linalg.inv(T_wc3)

    T_31_gt = T_c3_w_GT @ T_wc1
    R31_gt = T_31_gt[0:3,0:3]
    t31_gt = T_31_gt[0:3,3]

    translation_error = np.linalg.norm(t31 - t31_gt)
    print("Translation error:", translation_error)

    R_err = R31 @ R31_gt.T
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0) 
    angle_error = np.arccos(cos_angle)
    angle_error_deg = np.degrees(angle_error)
    print("Rotation error (degrees):", angle_error_deg)
    T_13_pnp = np.linalg.inv(T_31)
    T_wc3_pnp = T_wc1 @ T_13_pnp


    ########## BUNDLE ADJUSTMENT RESULTS COMPARISON WITH GT ##########

    # Final 3D plot with optimized points comparison with GT
    # X_op is in camera 1 reference frame, convert to world frame
    X_opt_h_gt = T_wc1 @ np.vstack((X_opt, np.ones((1, nPoints))))
    X_opt_gt = X_opt_h_gt[0:3, :] / X_opt_h_gt[3, :]

    fig3D_BA = plt.figure(11)
    ax2 = plt.axes(projection='3d', adjustable='box')
    ax2.set_xlabel('X') 
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    drawRefSystem(ax2, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax2, T_wc1, '-', 'C1')
    drawRefSystem(ax2, T_wc2, '-', 'C2')
    drawRefSystem(ax2, T_wc3, '-', 'C3')
    drawRefSystem(ax2, T_wc3_pnp, '--', 'C3_PnP')


    #GT points
    ax2.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.', alpha=0.4, label='GT points')
    #Optimized points
    ax2.scatter(X_opt_gt[0, :], X_opt_gt[1, :], X_opt_gt[2, :], marker='.', alpha=0.4, label='Optimized points')
    
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    ax2.legend()
    plt.title('Comparison of GT 3D points and optimized 3D points after Bundle Adjustment')
    plt.show() 