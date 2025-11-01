#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: SIFT matching
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


base_path = "D:/Dev_Space/python/Lab3/"
def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1): 
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        d_best = dist[indexSort[0]]
        d_second_sort = dist[indexSort[1]]
        ratio = d_best / d_second_sort
        if ((ratio < distRatio) and (dist[indexSort[0]] < minDist)): 
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
        
        
    return matches

def compute_homography(x1, x2):
    """Estima H a partir de 4+ puntos usando SVD."""
    n = x1.shape[0]
    A = []
    for i in range(n):
        X, Y = x1[i, 0], x1[i, 1]
        x, y = x2[i, 0], x2[i, 1]
        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]




def homograhy4Ransac(x1, x2,iterations, ransac_threshold): 
    """RANSAC manual para estimar la homografía H."""
    best_H = None
    best_inliers = []
    max_votes = 0

    n_points = x1.shape[0]
    for i in range(iterations):
        # 1️⃣ Selecciona 4 puntos aleatorios
        idx = np.random.choice(n_points, 4, replace=False)
        H = compute_homography(x1[idx], x2[idx])

        # 2️⃣ Proyecta todos los puntos de x1
        x1_h = np.hstack([x1, np.ones((n_points, 1))])
        projected = (H @ x1_h.T).T
        projected = projected[:, :2] / projected[:, [2]]

        # 3️⃣ Calcula error
        errors = np.linalg.norm(x2 - projected, axis=1)

        # 4️⃣ Inliers
        inliers = errors < ransac_threshold
        votes = np.sum(inliers)

        # 5️⃣ Guarda mejor H
        if votes > max_votes:
            max_votes = votes
            best_inliers = inliers
            best_H = H

    # 6️⃣ Recalcula H final con los inliers
    H_final = compute_homography(x1[best_inliers], x2[best_inliers])
    return H_final, best_inliers

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = base_path + 'image1.png'
    path_image_2 = base_path + 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    #(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)

    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.7
    #Discutir lo de la distancia después
    minDist = 500
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
    
    H_final, best_inliers = homograhy4Ransac(srcPts, dstPts, 2000, 5.0)
    print("Homografía estimada:\n", H_final)
    print(f"Inliers encontrados: {np.sum(best_inliers)} / {len(best_inliers)}")

    h1, w1 = image_pers_1.shape[:2]
    corners = np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32)

    proj = (H_final @ np.hstack([corners, np.ones((4,1))]).T).T
    proj = proj[:, :2] / proj[:, [2]]

    img_proj = image_pers_2.copy()
    cv2.polylines(img_proj, [np.int32(proj)], True, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB))
    plt.title("Plano estimado por la homografía SIFT (RANSAC)")
    plt.axis("off")
    plt.show()

    inlier_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if best_inliers[i]]
    outlier_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if not best_inliers[i]]
    
    print(f"Total de inliers: {len(inlier_matches)}")
    print(f"Total de outliers: {len(outlier_matches)}")


    img_combined = cv2.drawMatches(
    image_pers_1, keypoints_sift_1,
    image_pers_2, keypoints_sift_2,
    inlier_matches[:155], None, matchColor=(0,255,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title("Inliers (verde) detectados por RANSAC")
    plt.axis("off")
    plt.show()

    img_combined = cv2.drawMatches(
    image_pers_1, keypoints_sift_1,
    image_pers_2, keypoints_sift_2,
    outlier_matches[:100], None, matchColor=(0,0,255),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title("Outliers (rojo) detectados por RANSAC")
    plt.axis("off")
    plt.show()


    npz = np.load("D:/Dev_Space/python/Lab3/output/image1_image2_matches.npz")

    mask = npz['matches'] > -1
    idxs = npz['matches'][mask]

    x1_sp = npz['keypoints0'][mask]   # puntos en imagen 1
    x2_sp = npz['keypoints1'][idxs]   # puntos en imagen 2
    

    H_final, best_inliers = homograhy4Ransac(x1_sp, x2_sp, 2000, 5.0)

    print("Homografía estimada:\n", H_final)
    print(f"Inliers encontrados: {np.sum(best_inliers)} / {len(best_inliers)}")

    h1, w1 = image_pers_1.shape[:2]
    corners = np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32)

    proj = (H_final @ np.hstack([corners, np.ones((4,1))]).T).T
    proj = proj[:, :2] / proj[:, [2]]

    img_proj = image_pers_2.copy()
    cv2.polylines(img_proj, [np.int32(proj)], True, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB))
    plt.title("Plano estimado por la homografía con SuperGlue (RANSAC)")
    plt.axis("off")
    plt.show()

    inlier_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if best_inliers[i]]
    outlier_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if not best_inliers[i]]
    
    print(f"Total de inliers: {len(inlier_matches)}")
    print(f"Total de outliers: {len(outlier_matches)}")


    img_combined = cv2.drawMatches(
    image_pers_1, keypoints_sift_1,
    image_pers_2, keypoints_sift_2,
    inlier_matches[:155], None, matchColor=(0,255,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title("Inliers (verde) detectados por RANSAC")
    plt.axis("off")
    plt.show()

    img_combined = cv2.drawMatches(
    image_pers_1, keypoints_sift_1,
    image_pers_2, keypoints_sift_2,
    outlier_matches[:100], None, matchColor=(0,0,255),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title("Outliers (rojo) detectados por RANSAC")
    plt.axis("off")
    plt.show()

