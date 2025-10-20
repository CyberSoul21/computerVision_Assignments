#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Laboratory 2, 1)Triangulation
#
# Date: 20 October 2025
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################


from labSession2.plotData_v2 import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv 

#img1 = cv.cvtColor(cv.imread("labSession2/image1.png"), cv.COLOR_BGR2RGB)
#img2 = cv.cvtColor(cv.imread("labSession2/image2.png"), cv.COLOR_BGR2RGB)

def project_points(P, X_3D):
    """
    Project 3D points into image using projection matrix P.
    X_3D: Nx3 array of 3D points
    Returns Nx2 array of pixel coordinates (u,v)
    """
    X_h = np.hstack((X_3D, np.ones((X_3D.shape[0], 1))))  # make homogeneous
    x_proj = (P @ X_h.T).T
    x_proj /= x_proj[:, [2]]
    return x_proj[:, :2]

def project(P, Xw_h):
    """Vectorized 3x4 * 4xN -> 3xN, then dehomogenize -> 2xN."""
    S = P @ Xw_h              # 3xN (s*u, s*v, s)
    uv = S[:2, :] / S[2:, :]  # divide each row by s
    print("uv dehomogenize: ")
    print(uv)
    return uv 


def triangulate_point(x1,x2,P1,P2):
    n_points = x1.shape[0]
    X_est = np.zeros((n_points, 3))
    
    u_1_i, v_1_i = x1
    u_2_i, v_2_i = x2

    for i in range(n_points):
        A = np.array([
            u_1_i[i]*P1[2, :] - P2[0,:],
            v_1_i[i]*P1[2, :] - P2[1,:],
            u_2_i[i]*P1[2, :] - P2[0,:],
            v_2_i[i]*P1[2, :] - P2[1,:],
        ])
        _,_, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X/X[3]
        X_est[i] = X[:3]

    return X_est

#de camara a mundo,
T_w_c1 = np.loadtxt('labSession2/T_w_c1.txt')
T_w_c2 = np.loadtxt('labSession2/T_w_c2.txt')
#Invert transformations (we need world -> camera)
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)
K_c = np.loadtxt('labSession2/K_c.txt')
I_3x3 = np.identity(3)          # 3x3 identity
col_zeros = np.zeros((3,1))     # column of zeros (3x1)
I_matrix = np.hstack((I_3x3, col_zeros))

#######################################################################################
#1. Point Triangulation
#**************************************************************************************
#**************************************************************************************
#1.1 Compute the projection matrices P1 and P2.
#P1: Image 1
#Projection Matrices P1 and P2 // P = K[T_w_c]
#print("#Projection Matrices P1 and P2 // P = K[T_w_c]")
P1 = K_c @ I_matrix @ T_c1_w
#print("P1: ")
#print(P1)
P2 = K_c @ I_matrix @ T_c2_w
#print("P2: ")
#print(P2)



if __name__ == '__main__':

    #Image 1
    x1 = np.loadtxt('labSession2/x1Data.txt')
    #Image 2
    x2 = np.loadtxt('labSession2/x2Data.txt')


    X = triangulate_point(x1,x2,P1,P2)
    print(X)

    #Plotting in 3D
    # ---------------------------------------------------------
    # Plot the resulting 3D points
    # ---------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # === Scatter of triangulated points ===
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
            c='r', marker='.', s=5, label='Triangulated Points')

    # === Plot camera centers ===
    C1_w = T_w_c1[0:3, 3]
    C2_w = T_w_c2[0:3, 3]
    ax.scatter(C1_w[0], C1_w[1], C1_w[2], color='blue', s=40, label='Camera 1')
    ax.scatter(C2_w[0], C2_w[1], C2_w[2], color='green', s=40, label='Camera 2')

    # === draw camera axes ===
    drawRefSystem(ax, np.eye(4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    # === Labels and aspect ratio ===
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Triangulated 3D Points and Camera Poses')
    ax.set_box_aspect([1, 1, 1])   # keep axis proportions
    ax.view_init(elev=20, azim=-60)

    plt.show()
    #**************************************************************************************


    #**************************************************************************************
    # Reproject triangulated 3D points
    x1_reproj = project_points(P1, X)
    x2_reproj = project_points(P2, X)

    # ---------------------------------------------------------
    # Combined plot of Image 1 and Image 2
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ----- Left: Image 1 -----
    axs[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
    axs[0].plot(x1[:,0], x1[:,1], 'rx', markersize=8, label='Original')
    axs[0].plot(x1_reproj[:,0], x1_reproj[:,1], 'go', markersize=4, label='Reprojected')
    for k in range(x1.shape[0]):
        axs[0].text(x1[k,0]+5, x1[k,1]+5, str(k), color='r', fontsize=8)
    axs[0].set_title('Image 1: Original (red) vs Reprojected (green)')
    axs[0].legend()

    # ----- Right: Image 2 -----
    axs[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
    axs[1].plot(x2[:,0], x2[:,1], 'rx', markersize=8, label='Original')
    axs[1].plot(x2_reproj[:,0], x2_reproj[:,1], 'go', markersize=4, label='Reprojected')
    for k in range(x2.shape[0]):
        axs[1].text(x2[k,0]+5, x2[k,1]+5, str(k), color='r', fontsize=8)
    axs[1].set_title('Image 2: Original (red) vs Reprojected (green)')
    axs[1].legend()

    plt.suptitle("Reprojection validation: both views")
    plt.tight_layout()
    plt.show()
    #**************************************************************************************







