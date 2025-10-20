#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Laboratory 2, 2)Fundamental Matrix
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
from triangulation import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv 
from functools import partial
import random
from fundamentalEvaluation import *

#**********************************************************************
# Loaded Data

base_path = "labSession2/"

F_21_test = np.loadtxt (base_path + "F_21_test.txt")

img1 = cv.cvtColor(cv.imread(base_path + "image1.png"), cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(cv.imread(base_path + "image2.png"), cv.COLOR_BGR2RGB)

T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 matrix: camera1 -> world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 matrix: camera2 -> world
K_c    = np.loadtxt(base_path + "K_c.txt")

x1 = np.loadtxt(base_path + "x1Data.txt").T
x2 = np.loadtxt(base_path + "x2Data.txt").T
#**********************************************************************

#**********************************************************************
# Methods

def draw_epipolar_line(img, l, color):
    a, b, c = l
    h, w = img.shape[:2]
    x0, y0 = 0, int(-c / b)
    x1, y1 = w, int(-(c + a*w) / b)
    img_line = img.copy()
    cv2.line(img_line, (x0, y0), (x1, y1), color, 1)
    return img_line

def onclick_image1(event,F_21):
    """Callback when user clicks on image 1."""
    if event.inaxes != ax1_click:
        return

    u1, v1 = event.xdata, event.ydata
    print(f"\nClicked point in Image 1: ({u1:.2f}, {v1:.2f})")

    ###########################################
    x1_h = np.array([u1, v1, 1.0])
    l0 = F_21 @ x1_h
    print("Epipolar line coefficients in Image 2:", l0)
    ###########################################
    img1_click = img1.copy()
    cv2.circle(img1_click, (int(u1), int(v1)), 6, (255, 0, 0), -1)
    img2_line = draw_epipolar_line(img2, l0, (255, 0, 0))

    fig2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    ax1b.imshow(img1_click)
    ax1b.set_title(f"Clicked Point in Image 1 ({u1:.1f}, {v1:.1f})")
    ax2b.imshow(img2_line)
    ax2b.set_title("Epipolar Line in Image 2 (l0 = F x1)")
    plt.show()

def fundamental_from_poses(T_c1_w, T_c2_w, K):
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
        #X = triangulate_point(x_1, x_2, P1, P2)        

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

#**********************************************************************



if __name__ == '__main__': 

    #######################################################################################
    #2. Fundamental Matrix
    #**************************************************************************************
    #2.1 Epipolar lines visualization

    # Set up figure for interaction
    fig, (ax1_click, ax2_click) = plt.subplots(1, 2, figsize=(14, 6))
    ax1_click.imshow(img1)
    ax1_click.set_title("Click on a point in Image 1")
    ax2_click.imshow(img2)
    ax2_click.set_title("Image 2 (Epipolar line will appear here)")
    cid = fig.canvas.mpl_connect('button_press_event', partial(onclick_image1, F_21=F_21_test))
    plt.show()

    #**************************************************************************************


    #**************************************************************************************
    #2.2 Fundamental matrix definition
    F_21 = fundamental_from_poses(T_c1_w, T_c2_w, K_c) #(world → camera)
    #F21 = fundamental_from_poses(T_w_c1, T_w_c2, K_c) #4x4 camera1 → world

    # Set up figure for interaction
    fig, (ax1_click, ax2_click) = plt.subplots(1, 2, figsize=(14, 6))
    ax1_click.imshow(img1)
    ax1_click.set_title("Click on a point in Image 1")
    ax2_click.imshow(img2)
    ax2_click.set_title("Image 2 (Epipolar line will appear here)")
    cid = fig.canvas.mpl_connect('button_press_event', partial(onclick_image1, F_21=F_21))
    plt.show()
    #**************************************************************************************


    #**************************************************************************************
    #2.3 Fundamental matrix linear estimation with eight point solution.

    # Draw a few example epipolar lines
    num_points = 8
    n = x1.shape[0]                     # number of points
    k = min(num_points, n)              # don’t ask for more than exist
    indices = random.sample(range(n), k)
    img1_draw, img2_draw = img1.copy(), img2.copy()

    for i in indices:
        x1_h = np.array([x1[i,0], x1[i,1], 1])
        x2_h = np.array([x2[i,0], x2[i,1], 1])
        l0 = F_21 @ x1_h     # line in image 2
        l1 = F_21.T @ x2_h   # line in image 1
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1_draw = draw_epipolar_line(img1_draw, l1, color)
        img2_draw = draw_epipolar_line(img2_draw, l0, color)
        cv2.circle(img1_draw, (int(x1[i,0]), int(x1[i,1])), 5, color, -1)
        cv2.circle(img2_draw, (int(x2[i,0]), int(x2[i,1])), 5, color, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img1_draw)
    ax1.set_title("Image 1 with Epipolar Lines (l1 = F^T x2)")
    ax2.imshow(img2_draw)
    ax2.set_title("Image 2 with Epipolar Lines (l0 = F x1)")
    plt.show()
    #**************************************************************************************


    #**************************************************************************************
    #2.4 Pose estimation from two views

    #Compute Essential Matrix
    E = essentialMatrix(F_21, K_c)
    print("Essential Matrix:\n", E)

    # Get the four (R,t) combinations
    solutions = decomposeEssentialMatrix(E)
    print(f"\nFound {len(solutions)} possible camera poses.")

    # Select the correct one by triangulation
    best_pose, best_points = selectCorrectPose(K_c, x1, x2, solutions)
    R_best, t_best = best_pose

    print("\nBest Rotation:\n", R_best)
    print("\nBest Translation:\n", t_best)

    #**************************************************************************************

    #**************************************************************************************
    #2.5 Results presentation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_points[:, 0], best_points[:, 1], best_points[:, 2], c='r', marker='.')
    ax.set_title("Triangulated 3D points (best pose)")
    plt.show()


    #Metrics
    #1) Mean reprojection error on image 1 and 2.
    X = triangulatePoints(x1,x2,P1,P2)
    x1_reproj = project_points(P1, X)
    x2_reproj = project_points(P2, X)
    e1 = np.linalg.norm(x1_reproj - x1, axis=1)
    e2 = np.linalg.norm(x2_reproj - x2, axis=1)
    print(f"Mean reprojection error: img1={e1.mean():.2f}px, img2={e2.mean():.2f}px")
    #2) We might want to know how well is F_12, how well the correspondences satisfy epipolar geometry.
    #We found out Sampson error (first-order geometric error)
    #https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf
    #we did this function with IA assitance: fundamentalEvaluation
    print("********************************************************************")
    report = evaluate_F(F_21, x1, x2)
    print("Report for F21: ")
    print(report)

    print("********************************************************************")

    report_test = evaluate_F(F_21_test, x1, x2)
    print("Report for F21_test: ")
    print(report_test)


    #**************************************************************************************












