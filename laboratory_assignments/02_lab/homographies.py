#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Laboratory 2, 3)Homographies
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

def computeHomographyFromPose(poseCamera1world, poseCamera2world, K_instrinsict, plane):
    """
    Compute the homography relating image1 and image2 through a given plane
    defined in camera1 coordinates.
    H = K (R - t n^T / d) K^-1
    """
    # Relative pose between cameras
    T_c2_c1 = np.linalg.inv(poseCamera2world) @ poseCamera1world
    R = T_c2_c1[:3, :3]
    t = T_c2_c1[:3, 3].reshape(3, 1)

    # Plane parameters
    n = plane[:3].reshape(3, 1)
    d = plane[3]

    # Homography matrix
    H = K_instrinsict @ (R - (t @ n.T) / d) @ np.linalg.inv(K_instrinsict)
    H /= H[2, 2]  # normalize scale
    return H

def visualizeHomographyTransfer(image1, image2, H):
    """
    Click points in Image 1, project them via Homography H onto Image 2, 
    and show both images side by side.
    """
    # --- Mostrar imagen 1 y pedir clics ---
    plt.figure(1)
    plt.imshow(image1)
    plt.title('Click on ground points (close window when done)')
    plt.axis('on')

    print("Click multiple points in Image 1 (press ENTER or close window when finished)")
    pts = plt.ginput(n=-1, timeout=0)  # similar to lab1 logic
    plt.close()

    pts = np.array(pts)
    print(f"Selected {len(pts)} points:\n{pts}")

    # --- Aplicar homografía ---
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts2_h = (H @ pts_h.T).T
    pts2_h /= pts2_h[:, [2]]
    pts2 = pts2_h[:, :2]

    # --- Visualización como en Lab1 ---
    plt.figure(1)
    plt.imshow(image1)
    plt.plot(pts[:, 0], pts[:, 1], '+g', markersize=10)
    plotNumberedImagePoints(pts.T, 'y', (10, 10))
    plt.title('Image 1 - Clicked Points')
    plt.draw()
    plt.waitforbuttonpress()  # Espera antes de pasar a la siguiente imagen

    plt.figure(2)
    plt.imshow(image2)
    plt.plot(pts2[:, 0], pts2[:, 1], '+r', markersize=10)
    plotNumberedImagePoints(pts2.T, 'yellow', (10, 10))
    plt.title('Image 2 - Projected Points via Homography')
    plt.draw()
    plt.waitforbuttonpress()

    print("Visualization complete. Close the figures to continue.")
    plt.show()


def normalize_points(xh):
    xh = xh / xh[:,2:3]
    u,v = xh[:,0], xh[:,1]
    uc, vc = u.mean(), v.mean()
    u0, v0 = u-uc, v-vc
    md = np.mean(np.sqrt(u0**2 + v0**2))
    s = np.sqrt(2)/md if md>0 else 1.0
    T = np.array([[s,0,-s*uc],[0,s,-s*vc],[0,0,1]], float)
    return (T @ xh.T).T, T

def estimate_H(x1, x2):
    x1h = np.hstack([x1, np.ones((x1.shape[0],1))])
    x2h = np.hstack([x2, np.ones((x2.shape[0],1))])
    print(x1.shape)    # (N, 2)
    print(x1h.shape)   # (N, 3)
    # Normalize
    x1n, T1 = normalize_points(x1h.transpose())
    x2n, T2 = normalize_points(x2h.transpose())

    A = []
    for (x, y), (u, v) in zip(x1n[:,:2], x2n[:,:2]):
        A.append([ 0, 0, 0, -x, -y, -1, v*x, v*y, v ])
        A.append([ x, y, 1,  0,  0,  0, -u*x, -u*y, -u ])
    A = np.array(A)

    # Solve A h = 0
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)

    # Denormalize
    H = np.linalg.inv(T2) @ Hn @ T1
    H /= H[2,2]
    return H    

def reprojection_errors(H, x1, x2):
    x1h = np.hstack([x1, np.ones((x1.shape[0],1))]).transpose()
    x2 = np.hstack([x1, np.ones((x2.shape[0],1))]).transpose()
    x2_proj = (H @ x1h.T).T
    x2_proj = x2_proj[:, :2] / x2_proj[:, 2:3]
    errors = np.linalg.norm(x2[:, :2] - x2_proj, axis=1)
    return errors


if __name__ == '__main__': 
    #######################################################################################
    #3. Homographies
    #**************************************************************************************
    #3.1 Homography definition  
    # compute the homography that relates both images through the floor plane 
    # Plane equation in camera1 reference: π1 = [n_x, n_y, n_z, d]
    pi = np.array([0.0149, 0.9483, 0.3171, -1.7257])
    H_from_pose = computeHomographyFromPose(T_w_c1, T_w_c2, K_c, pi)

    #**************************************************************************************

    #**************************************************************************************
    #3.2 Point transfer visualization   

    #visualizeHomographyTransfer(img1, img2, H_from_pose)

    #**************************************************************************************

    #**************************************************************************************
    #3.3 Homography linear estimation from matches  


    # Load Data
    folder = "labSession2/"
    x1 = np.loadtxt(folder + "x1FloorData.txt")  # image 1
    x2 = np.loadtxt(folder + "x2FloorData.txt")  # image 2

    # Estimate H using direct linear trasnform 
    H_est = estimate_H(x1, x2)
    print("Estimated Homography H_21:\n", H_est)

    # Metricts
    errors = reprojection_errors(H_est, x1, x2)
    print(f"Mean reprojection error: {errors.mean():.2f} px")
    print(f"Median reprojection error: {np.median(errors):.2f} px")

    errors = reprojection_errors(H_from_pose, x1, x2)
    print(f"Mean reprojection error: {errors.mean():.2f} px")
    print(f"Median reprojection error: {np.median(errors):.2f} px")


    #**************************************************************************************