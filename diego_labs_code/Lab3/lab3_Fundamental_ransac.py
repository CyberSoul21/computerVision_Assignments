#####################################################################################
# MRGCV Unizar - Laboratory 3
# Title: Fundamental Matrix + Automatic Epipolar Lines Visualization
# 
#####################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==========================================================
# Fundamental matrix (8-point algorithm)
# ==========================================================
def computeFundamental(x1, x2):
    def normalize_points(points):
        mean = np.mean(points, axis=0)
        centered = points - mean
        scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(centered**2, axis=1)))
        T = np.array([[scale, 0, -scale * mean[0]],
                    [0, scale, -scale * mean[1]],
                    [0, 0, 1]])
        pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
        pts_n = (T @ pts_h.T).T[:, :2]
        return pts_n, T

       # Normalize
    x1n, T1 = normalize_points(x1)
    x2n, T2 = normalize_points(x2)
    n = x1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        X, Y = x1n[i]
        x, y = x2n[i]
        A[i] = [X*x, X*y, X, Y*x, Y*y, Y, x, y, 1]
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


# ==========================================================
# RANSAC estimation of Fundamental matrix
# ==========================================================
def ransacFundamental(x1, x2, iterations, threshold):
    best_F, best_inliers = None, None
    max_votes = 0
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])

    for _ in range(iterations):
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        F = computeFundamental(x1[idx], x2[idx])
        Fx1 = (F @ x1_h.T).T
        Ftx2 = (F.T @ x2_h.T).T
        x2tFx1 = np.sum(x2_h * Fx1, axis=1)
        error = (x2tFx1 ** 2) / (Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2)

        inliers = error < threshold
        votes = np.sum(inliers)

        if votes > max_votes:
            max_votes = votes
            best_F = F
            best_inliers = inliers

    F_final = computeFundamental(x1[best_inliers], x2[best_inliers])
    return F_final, best_inliers


# ==========================================================
# Draw automatic epipolar lines using inlier matches
# ==========================================================

def compute_epipolar_lines(F, pts1, pts2, img1, img2, n_lines=15):
    """
    Dibuja líneas epipolares de manera explícita (sin usar OpenCV).
    Para cada punto x1 de la imagen 1, se calcula l2 = F * x1.
    """

    # Convertir imágenes a color
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h, w = img2.shape[:2]

    # Seleccionar subconjunto aleatorio de puntos
    idx = np.random.choice(len(pts1), min(n_lines, len(pts1)), replace=False)
    pts1_sel = pts1[idx]
    pts2_sel = pts2[idx]

    for pt1, pt2 in zip(pts1_sel, pts2_sel):
        # Convertir a coordenadas homogéneas
        x1 = np.array([pt1[0], pt1[1], 1.0])
        # Línea epipolar en imagen 2: l2 = F * x1
        l2 = F @ x1
        norm = np.sqrt(l2[0]**2 + l2[1]**2)
        l2 /= norm
        a, b, c = l2

        # Evitar divisiones por cero
        if abs(b) > 1e-8:
            x_vals = np.array([0, w])
            y_vals = -(a * x_vals + c) / b
        else:
            x_vals = -c / a * np.ones(2)
            y_vals = np.array([0, h])

        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Dibuja punto en imagen 1
        img1_color = cv2.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)

        # Dibuja la línea epipolar en imagen 2
        img2_color = cv2.line(img2_color,
                              (int(x_vals[0]), int(y_vals[0])),
                              (int(x_vals[-1]), int(y_vals[-1])),
                              color, 1)
        # Punto correspondiente en imagen 2
        img2_color = cv2.circle(img2_color, tuple(np.int32(pt2)), 4, color, -1)

    return img1_color, img2_color

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

# ==========================================================
# Load matches (SIFT or SuperGlue)
# ==========================================================
def load_matches(mode="sift"):
    base_path = "D:/Dev_Space/python/Lab3/"
    img1 = cv2.imread(base_path + "image1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(base_path + "image2.png", cv2.IMREAD_GRAYSCALE)

    if mode == "sift":
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
        kpts1, desc1 = sift.detectAndCompute(img1, None)
        kpts2, desc2 = sift.detectAndCompute(img2, None)
        
        
        distRatio = 0.7
        #Discutir lo de la distancia después
        minDist = 500
        matchesList = matchWith2NDRR(desc1, desc2, distRatio, minDist)
        dMatchesList = indexMatrixToMatchesList(matchesList)
        dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
        matchesList = matchesListToIndexMatrix(dMatchesList)

        # Matched points in numpy from list of DMatches
        srcPts = np.float32([kpts1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        dstPts = np.float32([kpts2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        print(f"[SIFT] Matches: {len(dMatchesList)}")
        return img1, img2, srcPts, dstPts

    elif mode == "superglue":
        npz = np.load(base_path + "image1_image2_matches.npz")
        mask = npz["matches"] > -1
        idxs = npz["matches"][mask]
        srcPts = npz["keypoints0"][mask]
        dstPts = npz["keypoints1"][idxs]
        print(f"[SuperGlue] Matches: {len(srcPts)}")
        return img1, img2, srcPts, dstPts


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    # Choose between "sift" or "superglue"
    mode = "superglue"  # or "superglue"

    img1, img2, srcPts, dstPts = load_matches(mode)
    F, inliers = ransacFundamental(srcPts, dstPts, iterations=5000, threshold=0.00002)
    print("Fundamental Matrix:\n", F)
    print(f"Inliers: {np.sum(inliers)} / {len(inliers)}")

    inlier_pts1 = srcPts[inliers]
    inlier_pts2 = dstPts[inliers]

    img1_epi, img2_epi = compute_epipolar_lines(
    F, inlier_pts1, inlier_pts2,
    img1, img2, len(inliers))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1_epi, cv2.COLOR_BGR2RGB))
    plt.title(f"Epipolar Lines in Image 1 ({mode.upper()})")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2_epi, cv2.COLOR_BGR2RGB))
    plt.title(f"Epipolar Lines in Image 2 ({mode.upper()})")
    plt.axis("off")

    plt.show()

    print("Rank(F):", np.linalg.matrix_rank(F))
    print("F normalized:\n", F / np.linalg.norm(F))

