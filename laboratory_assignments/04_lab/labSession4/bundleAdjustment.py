import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scOptim
import scipy.linalg as scl
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # just for 3D plots


basePath = "labSession4/"
# ---------- Existing helpers from your Lab 2 ----------

def project_points(P, X_3D):
    X_h = np.hstack((X_3D, np.ones((X_3D.shape[0], 1))))  # Nx4
    x_proj = (P @ X_h.T).T                                # Nx3
    x_proj /= x_proj[:, [2]]
    return x_proj[:, :2]

def essentialMatrix(fundamentalMatrix, K_intrinsic):
    e_Matrix = K_intrinsic.T @ fundamentalMatrix @ K_intrinsic
    U, S, Vt = np.linalg.svd(e_Matrix)
    S = [1, 1, 0]
    e_Matrix = U @ np.diag(S) @ Vt
    return e_Matrix

def decomposeEssentialMatrix(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    solutions = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]
    return solutions

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

def selectCorrectPose(K, x_1, x_2, possible_solutions):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_count = 0
    best_pose = None
    best_points = None

    for i, (R, t) in enumerate(possible_solutions):
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangulatePoints(x_1, x_2, P1, P2)

        X_cam2 = (R @ X.T + t.reshape(3, 1)).T

        count = np.sum((X[:, 2] > 0) & (X_cam2[:, 2] > 0))
        print(f"Solution {i+1}: {count} points with positive depth")

        if count > best_count:
            best_count = count
            best_pose = (R, t)
            best_points = X

    print(f"\nBest solution: {best_count} points in front of both cameras.")
    return best_pose, best_points

# ---------- New SO(3) helpers for Lab 4 ----------

def crossMatrix(x):
    """Skew-symmetric matrix [x]_x for a 3-vector x."""
    return np.array([[0,     -x[2],  x[1]],
                     [x[2],   0,    -x[0]],
                     [-x[1],  x[0],  0   ]], dtype=float)

def crossMatrixInv(M):
    """Inverse of crossMatrix: from skew-symmetric matrix to vector."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)

def so3_exp(theta):
    """Exponential map: 3-vector -> 3x3 rotation matrix."""
    return scl.expm(crossMatrix(theta))

def so3_log(R):
    """Logarithmic map: 3x3 rotation -> 3-vector."""
    M = scl.logm(R.astype('float64'))
    M = M.real  # discard tiny imaginary part if present
    return crossMatrixInv(M)

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Residuals for two-view bundle adjustment.

    Op: [theta(3), t(3), X1(3*nPoints)]
    x1Data, x2Data: 3 x nPoints homogeneous image points
    K_c: 3 x 3 intrinsics
    """
    # ---- unpack parameters ----
    theta = Op[0:3]
    t = Op[3:6]
    X_flat = Op[6:]
    X1 = X_flat.reshape((nPoints, 3))

    # rotation from 1 to 2
    R_21 = so3_exp(theta)

    # Projection matrices
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))         # K [I|0]
    P2 = K_c @ np.hstack((R_21, t.reshape(3, 1)))               # K [R|t]

    # Measured points in pixel coordinates (Nx2)
    x1_obs = (x1Data[0:2, :] / x1Data[2, :]).T
    x2_obs = (x2Data[0:2, :] / x2Data[2, :]).T

    # Predicted projections
    x1_proj = project_points(P1, X1)    # Nx2
    x2_proj = project_points(P2, X1)    # Nx2

    # Residuals: measured - projected (u,v for both cameras)
    res1 = (x1_obs - x1_proj).ravel()
    res2 = (x2_obs - x2_proj).ravel()

    res = np.hstack((res1, res2))
    return res


def plot_3d_results(R_est, t_est, X1_est, T_w_c1, T_w_c2, X_w):
    """
    3D visualization of cameras and points:
    - GT cameras and points in world frame (green + red/magenta)
    - Estimated camera 2 and points in world frame (blue + cyan)
    """

    # Ground-truth points in world
    Xw_gt = X_w[:3, :].T   # N x 3

    # Ground-truth camera centers in world
    C1_w = (T_w_c1 @ np.array([0, 0, 0, 1.0]))[:3]
    C2_w = (T_w_c2 @ np.array([0, 0, 0, 1.0]))[:3]

    # Estimated camera 2 center:
    # first in camera-1 frame, then to world using T_w_c1
    C2_c1 = -R_est.T @ t_est
    C2_est_w = (T_w_c1 @ np.hstack((C2_c1, 1.0)))[:3]

    # Estimated 3D points: from cam1 frame to world
    X1_est_w = (T_w_c1 @
                np.hstack((X1_est, np.ones((X1_est.shape[0], 1)))).T).T[:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D scene: ground truth vs BA estimate")

    # Points
    ax.scatter(Xw_gt[:, 0], Xw_gt[:, 1], Xw_gt[:, 2],
               c='g', marker='.', label='GT points')
    ax.scatter(X1_est_w[:, 0], X1_est_w[:, 1], X1_est_w[:, 2],
               c='b', marker='.', label='Est points')

    # Cameras
    ax.scatter(C1_w[0], C1_w[1], C1_w[2],
               c='r', marker='^', s=60, label='GT cam 1')
    ax.scatter(C2_w[0], C2_w[1], C2_w[2],
               c='m', marker='^', s=60, label='GT cam 2')
    ax.scatter(C2_est_w[0], C2_est_w[1], C2_est_w[2],
               c='c', marker='^', s=60, label='Est cam 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=120, suppress=True)

    # ----- Load data -----
    K_c = np.loadtxt(basePath + "K_c.txt")
    F_21 = np.loadtxt(basePath + "F_21.txt")

    x1 = np.loadtxt(basePath + "x1Data.txt")   # shape (2, N)
    x2 = np.loadtxt(basePath + "x2Data.txt")   # shape (2, N)

    T_w_c1 = np.loadtxt(basePath + "T_w_c1.txt")
    T_w_c2 = np.loadtxt(basePath + "T_w_c2.txt")
    X_w = np.loadtxt(basePath + "X_w.txt")   # 4 x N (homogeneous)    

    # >>> load images <<<
    img1 = mpimg.imread(basePath + "image1.png")   # change name if needed
    img2 = mpimg.imread(basePath + "image2.png")

    # Make them 3xN homogeneous
    x1Data = np.vstack((x1, np.ones((1, x1.shape[1]))))
    x2Data = np.vstack((x2, np.ones((1, x2.shape[1]))))
    nPoints = x1Data.shape[1]

    print("Loaded", nPoints, "point correspondences.")

    # ----- Initial solution from essential matrix (Lab 2) -----
    E = essentialMatrix(F_21, K_c)
    sols = decomposeEssentialMatrix(E)

    # x_1, x_2 for selectCorrectPose: Nx2 pixel coords
    x1_px = x1Data[0:2, :].T
    x2_px = x2Data[0:2, :].T

    (R_21_init, t_21_init), X1_init = selectCorrectPose(K_c, x1_px, x2_px, sols)

    print("Initial R_21:\n", R_21_init)
    print("Initial t_21:\n", t_21_init)
    print("First 3 initial 3D points:\n", X1_init[:3])

    # ----- Build initial optimization vector Op0 -----
    theta0 = so3_log(R_21_init)
    Op0 = np.zeros(6 + 3 * nPoints)
    Op0[0:3] = theta0
    Op0[3:6] = t_21_init
    Op0[6:] = X1_init.reshape(-1)

    # ----- Test residuals with initial solution -----
    res0 = resBundleProjection(Op0, x1Data, x2Data, K_c, nPoints)
    print("Initial residual norm:", np.linalg.norm(res0))

    # Visualize initial projections vs measurements
    def plot_projections(R_21, t_21, X1, title_prefix="Initial"):
        P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K_c @ np.hstack((R_21, t_21.reshape(3, 1)))

        x1_proj = project_points(P1, X1).T    # 2xN
        x2_proj = project_points(P2, X1).T

        plt.figure()
        plt.title(f"{title_prefix} - Image 1")
        plt.scatter(x1Data[0, :], x1Data[1, :], c='r', marker='x', label='measured')
        plt.scatter(x1_proj[0, :], x1_proj[1, :],
                    facecolors='none', edgecolors='b', label='projected')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.legend()

        plt.figure()
        plt.title(f"{title_prefix} - Image 2")
        plt.scatter(x2Data[0, :], x2Data[1, :], c='r', marker='x', label='measured')
        plt.scatter(x2_proj[0, :], x2_proj[1, :],
                    facecolors='none', edgecolors='b', label='projected')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.legend()

    def plot_projections(R_21, t_21, X1, img1, img2, title_prefix="Initial"):
        # Projection matrices
        P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K_c @ np.hstack((R_21, t_21.reshape(3, 1)))

        # Predicted projections
        x1_proj = project_points(P1, X1).T   # 2xN
        x2_proj = project_points(P2, X1).T

        # ---------- Image 1 ----------
        plt.figure()
        plt.title(f"{title_prefix} - Image 1")
        plt.imshow(img1, cmap='gray')   # background image

        # measured points
        plt.scatter(x1Data[0, :], x1Data[1, :],
                    c='r', marker='x', label='measured')

        # projected points
        plt.scatter(x1_proj[0, :], x1_proj[1, :],
                    facecolors='none', edgecolors='b', label='projected')

        plt.axis('image')
        plt.xlim(0, img1.shape[1])
        plt.ylim(img1.shape[0], 0)   # y downwards, like image coords
        plt.legend()

        # ---------- Image 2 ----------
        plt.figure()
        plt.title(f"{title_prefix} - Image 2")
        plt.imshow(img2, cmap='gray')

        plt.scatter(x2Data[0, :], x2Data[1, :],
                    c='r', marker='x', label='measured')
        plt.scatter(x2_proj[0, :], x2_proj[1, :],
                    facecolors='none', edgecolors='b', label='projected')

        plt.axis('image')
        plt.xlim(0, img2.shape[1])
        plt.ylim(img2.shape[0], 0)
        plt.legend()


    #plot_projections(R_21_init, t_21_init, X1_init, title_prefix="Initial")
    plot_projections(R_21_init, t_21_init, X1_init, img1, img2, title_prefix="Initial")
    print("Close the figures to continue to BA.")
    plt.show()

    # ----- Bundle Adjustment using least squares -----
    print("Running bundle adjustment (two views)...")
    OpOptim = scOptim.least_squares(
        resBundleProjection,
        Op0,
        args=(x1Data, x2Data, K_c, nPoints),
        method='lm'
    )

    print("Optimization success:", OpOptim.success)
    print("Final cost:", OpOptim.cost)

    # ----- Extract refined parameters -----
    theta_opt = OpOptim.x[0:3]
    t_opt = OpOptim.x[3:6]
    X1_opt = OpOptim.x[6:].reshape((nPoints, 3))

    R_opt = so3_exp(theta_opt)

    print("Refined R_21:\n", R_opt)
    print("Refined t_21:\n", t_opt)

    # Residual norm after BA
    res_final = resBundleProjection(OpOptim.x, x1Data, x2Data, K_c, nPoints)
    print("Final residual norm:", np.linalg.norm(res_final))

    # Visualize refined projections
    #plot_projections(R_opt, t_opt, X1_opt, title_prefix="Refined (BA)")
    plot_projections(R_opt, t_opt, X1_opt, img1, img2, title_prefix="Refined (BA)")
    print("BA finished. Inspect residual plots and 3D structure.")
    plt.show()


    theta_opt = OpOptim.x[0:3]
    t_opt = OpOptim.x[3:6]
    X1_opt = OpOptim.x[6:].reshape((nPoints, 3))    
    R_opt = so3_exp(theta_opt)
    # 3D visualization: cameras + points vs ground truth
    plot_3d_results(R_opt, t_opt, X1_opt, T_w_c1, T_w_c2, X_w)


    input()
