import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------
# Functions predefined
# ---------------------------------------------------------------


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]): 
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


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
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

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


#Define multiplication of matrices
def multi_M(A,B): 
    if A.shape[1] != B.shape[0]:
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B.")
    return A @ B


# ---------------------------------------------------------------
# Defintion of trinagulation
# ---------------------------------------------------------------

def triangulate_point (x1,x2,P_camera1,P_camera2): 
    A = np.array([
        x1[0]*P_camera1[2, :] - P_camera1[0,:],
        x1[1]*P_camera1[2, :] - P_camera1[1,:],
        x2[0]*P_camera2[2, :] - P_camera2[0,:],
        x2[1]*P_camera2[2, :] - P_camera2[1,:],
    ])

    _,_, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X/X[3]
    return X[:3]
    
# ---------------------------------------------------------
# 1. Define base path for your Lab files
# ---------------------------------------------------------
base_path = "D:/Dev_Space/python/Lab2/"


# ---------------------------------------------------------
# 5. Load matched 2D points (each file has 2 rows: u and v)
# ---------------------------------------------------------
x1 = np.loadtxt(base_path + "x1Data.txt").T
x2 = np.loadtxt(base_path + "x2Data.txt").T

  # ---------------------------------------------------------
# 2. Load extrinsic and intrinsic parameters
# ---------------------------------------------------------
T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 matrix: camera1 -> world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 matrix: camera2 -> world
K_c    = np.loadtxt(base_path + "K_c.txt")     # 3x3 intrinsic matrix


#-----------------------------------------------
# Load images
#----------------------------------------------
img1 = cv.cvtColor(cv.imread(base_path + "image1.png"), cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(cv.imread(base_path + "image2.png"), cv.COLOR_BGR2RGB)


if __name__ == '__main__':
    
    # ---------------------------------------------------------
    # 3. Invert transformations (we need world -> camera)
    # ---------------------------------------------------------
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)

    # ---------------------------------------------------------
    # 4. Build projection matrices P = K [R | t]
    # ---------------------------------------------------------
    R1 = T_c1_w[0:3, 0:3]
    t1 = T_c1_w[0:3, 3].reshape(3, 1)
    R2 = T_c2_w[0:3, 0:3]
    t2 = T_c2_w[0:3, 3].reshape(3, 1)

    P1 = multi_M(K_c,np.hstack((R1, t1)))
    P2 = multi_M(K_c,np.hstack((R2,t2)))
    
    #Triangulation of points
    n_points = x1.shape[0]
    X_est = np.zeros((n_points, 3))

    for i in range(n_points):
        X_est[i] = triangulate_point(x1[i], x2[i], P1, P2)
    


    ###Aid by Chatgpt
    #Plotting in 3D
    # ---------------------------------------------------------
    # 8. Plot the resulting 3D points (clean version)
    # ---------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # === Scatter of triangulated points ===
    ax.scatter(X_est[:, 0], X_est[:, 1], X_est[:, 2],
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
    #Chat GPT aid
    # ---------------------------------------------------------
    # 9. Combined 2D Plotting for both images
    # ---------------------------------------------------------

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

    # Reproject triangulated 3D points
    x1_reproj = project_points(P1, X_est)
    x2_reproj = project_points(P2, X_est)

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