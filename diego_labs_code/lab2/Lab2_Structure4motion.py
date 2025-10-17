import matplotlib.pylab as plt 
import numpy as np 
import cv2 as cv

#-------------------------------
# Loaded Documents

base_path = "D:/Dev_Space/python/Lab2/"

f_matrix_test = np.loadtxt (base_path + "F_21_test.txt")

img1 = cv.cvtColor(cv.imread(base_path + "image1.png"), cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(cv.imread(base_path + "image2.png"), cv.COLOR_BGR2RGB)

T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 matrix: camera1 -> world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 matrix: camera2 -> world
K_c    = np.loadtxt(base_path + "K_c.txt")

x1 = np.loadtxt(base_path + "x1Data.txt").T
x2 = np.loadtxt(base_path + "x2Data.txt").T

# ------------------------------
# defined functions

def homogeneousPoint(x): 
    if isinstance(x, list): 
        x = x[0]
        x = np.asarray(x)
    return np.append(x, 1.0)

###Pedir ayuda al profesor
def computeEpipolarLine(x1, F):
    """
    Given a 2D point x1 (in homogeneous coords) in image 1 and fundamental matrix F,
    compute the corresponding epipolar line l2 = F * x1 in image 2.
    Returns normalized (a, b, c).
    """
    l2 = F @ x1
    norm = np.sqrt(l2[0]**2 + l2[1]**2)
    
    if norm > 1e-8:
        l2 /= norm
    return l2  # [a, b, c]

###ChatGptAided
def plotEpipolarGeometry(img1, img2, x1, l2):
    a, b, c = l2
    h, w = img2.shape[:2]

    # Compute y for x limits
    if abs(b) > 1e-8:
        x_vals = np.array([0, w])
        y_vals = -(a * x_vals + c) / b
    else:
        # vertical line
        x_vals = -c / a * np.ones(2)
        y_vals = np.array([0, h])

    # Plot side-by-side comparison
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Image 1 with clicked point
    ax[0].imshow(img1)
    ax[0].plot(x1[0], x1[1], 'go', markersize=10)
    ax[0].set_title("Image 1 (clicked point)")
    ax[0].axis('on')

    # Right: Image 2 with epipolar line
    ax[1].imshow(img2)
    ax[1].plot(x_vals, y_vals, 'r-', linewidth=2)
    ax[1].set_title("Image 2 (epipolar line)")
    ax[1].axis('on')

    plt.show()



def l_epipolar (image1, image2, fundamentalMatrix): 
    #Showing the first image
    plt.figure(figsize=(8,6))
    plt.imshow(image1)
    plt.title ("Image 1 - Camera View Click on any point on the image")
    plt.axis('on')
    # Wait for ONE click from the user
    print("Click a point on the image and close the window...")
    clicked_points = plt.ginput(1)  # waits for one click
    plt.close()
    #make it homogeneos
    extractPoint = homogeneousPoint(clicked_points)
    #Check chape should be (3,) 
    print (extractPoint.shape)
    # Compute line
    line = computeEpipolarLine(extractPoint, fundamentalMatrix)
    print("Epipolar line coefficients (a,b,c):", line)

    # Plot geometry
    plotEpipolarGeometry(image1, image2, extractPoint, line)


#-----------------------------
#Calculaiting Fundamental Matrix
def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

    
def fundMatrix(poseWorld2CameraOne, poseWorld2CameraTwo, K_intrinsict):
    #poseCameraOne2world = np.linalg.inv(poseWorld2CameraOne)
    poseCameraTwo2world = np.linalg.inv(poseWorld2CameraTwo)
    p_cOne2cTwo = poseCameraTwo2world @ poseWorld2CameraOne
    R = p_cOne2cTwo [0:3, 0:3]
    t = p_cOne2cTwo [0:3, 3]
    essentialMatrix = skew(t) @ R
    ##hasta aqui entiendo
    K_inverse = np.linalg.inv(K_intrinsict)
    fundaMatrix = K_inverse.T @ essentialMatrix @ K_inverse
    fundaMatrix = fundaMatrix/np.linalg.norm(fundaMatrix) 
    return fundaMatrix
    
def fundMatrixFromPoints(x1_points , x2_points): 
    """
    Estimate Fundamental Matrix F using the normalized 8-point algorithm.
    Input:
        x1, x2 : Nx2 arrays of corresponding points (u,v)
    Output:
        F : 3x3 Fundamental matrix
    """
    n = x1_points.shape[0]
    if n < 8: 
        raise  ValueError("At least we need 8 points")
    
    A = np.zeros((n,9))
    for i in range(n): 
        u1, v1 = x1_points[i, 0], x1_points[i, 1]
        u2, v2 = x2_points[i, 0], x2_points[i, 1]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    #solving A with svd
    _,_,Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3 , 3)

    # Enforce rank 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Normalize
    F /= np.linalg.norm(F)
    return F

def essentialMatrix(fundamentalMatrix, K_intrinsict):
    e_Matrix = K_intrinsict.T @ fundamentalMatrix @ K_intrinsict
    U, S, Vt = np.linalg.svd(e_Matrix)
    S = [1,1,0]
    e_Matrix = U @ np.diag(S) @ Vt
    return e_Matrix

def decomposeEssentialMatrix(essentialMatrix): 
    U, _, Vt = np.linalg.svd(essentialMatrix)

    # Ensure a proper rotation (det(R) = +1)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # 4 possible solutions
    solutions = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]
    return solutions


def triangulatePoints (x_1,x_2,P1,P2): 
    """
    Triangulate corresponding points between two images.
    Returns Nx3 array of 3D points (in homogeneous coordinates normalized).
    """
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
    """
    Evaluate the 4 (R,t) combinations and return the physically correct one.
    Criterion: Most triangulated points have positive depth (Z>0) in both cameras.
    """
    # First camera at origin
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_count = 0
    best_pose = None
    best_points = None

    for i, (R, t) in enumerate(possible_solutions):
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangulatePoints(x_1, x_2, P1, P2)

        # Check positive depth
        X_h = np.hstack((X, np.ones((X.shape[0], 1))))
        X_cam2 = (R @ X.T + t.reshape(3, 1)).T

        count = np.sum((X[:, 2] > 0) & (X_cam2[:, 2] > 0))
        print(f"Solution {i+1}: {count} points with positive depth")

        if count > best_count:
            best_count = count
            best_pose = (R, t)
            best_points = X

    print(f"\n Best solution: {best_count} points in front of both cameras.")
    return best_pose, best_points



if __name__ == '__main__': 
    #print(x1)
    #fundamentalMatrix_Lab = fundMatrix(T_w_c1,T_w_c2,K_c)
    #fundamentalMatrix_fromPoints =fundMatrixFromPoints(x1,x2)
    #l_epipolar(img1, img2, fundamentalMatrix_Lab)
    #l_epipolar(img1, img2, fundamentalMatrix_fromPoints)
    
     #1️⃣ Compute Essential Matrix
    E = essentialMatrix(f_matrix_test, K_c)
    print("Essential Matrix:\n", E)

    # 2️⃣ Get the four (R,t) combinations
    solutions = decomposeEssentialMatrix(E)
    print(f"\nFound {len(solutions)} possible camera poses.")

    # 3️⃣ Select the correct one by triangulation
    best_pose, best_points = selectCorrectPose(K_c, x1, x2, solutions)
    R_best, t_best = best_pose

    print("\nBest Rotation:\n", R_best)
    print("\nBest Translation:\n", t_best)

    # Optional: visualize points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_points[:, 0], best_points[:, 1], best_points[:, 2], c='r', marker='.')
    ax.set_title("Triangulated 3D points (best pose)")
    plt.show()

    



