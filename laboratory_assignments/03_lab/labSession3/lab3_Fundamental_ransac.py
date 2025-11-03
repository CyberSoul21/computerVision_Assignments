#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Laboratory 3, 5)Fundamental Matrix + Automatic Epipolar Lines Visualization
# Date: 03 November 2025
#
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
#          Diego Mendez, 960616
#
#####################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==========================================================
# Fundamental matrix (8-point algorithm)
# ==========================================================
def computeFundamental_Diego(x1, x2):
    #Normalizacion de Hartley
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

def computeFundamental(pts1, pts2):
    import numpy as np
    def normalize(xy):
        m = xy.mean(axis=0)
        d = np.sqrt(((xy - m)**2).sum(axis=1)).mean()
        s = np.sqrt(2)/(d+1e-12)
        T = np.array([[s,0,-s*m[0]],[0,s,-s*m[1]],[0,0,1]])
        xh = np.hstack([xy, np.ones((xy.shape[0],1))]).T
        xn = (T @ xh); xn = (xn[:2]/xn[2]).T
        return xn, T

    x1n, T1 = normalize(pts1)
    x2n, T2 = normalize(pts2)

    x, y = x1n[:,0], x1n[:,1]
    u, v = x2n[:,0], x2n[:,1]
    A = np.vstack([u*x, u*y, u, v*x, v*y, v, x, y, np.ones_like(x)]).T

    _, _, Vt = np.linalg.svd(A)
    Fh = Vt[-1].reshape(3,3)

    U,S,Vt = np.linalg.svd(Fh)
    S[-1] = 0.0
    Fh = U @ np.diag(S) @ Vt

    F = T2.T @ Fh @ T1
    return F / (np.linalg.norm(F) + 1e-12)


# ==========================================================
# RANSAC estimation of Fundamental matrix
# ==========================================================
def ransacFundamental(x1, x2, iterations, threshold):
    best_F, best_inliers = None, None
    max_votes = 0

    # Puntos homogéneos
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])

    for _ in range(iterations):
        # Seleccionar 8 puntos aleatorios
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        F = computeFundamental(x1[idx], x2[idx])

        # Líneas epipolares correspondientes
        Fx1 = (F @ x1_h.T).T      # Líneas en imagen 2
        Ftx2 = (F.T @ x2_h.T).T   # Líneas en imagen 1

        # Término común de la ecuación epipolar
        x2tFx1 = np.sum(x2_h * Fx1, axis=1)

        # ---- ERROR GEOMÉTRICO CUADRÁTICO ----
        # Distancia perpendicular en ambas imágenes
        d1 = np.abs(x2tFx1) / np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
        d2 = np.abs(x2tFx1) / np.sqrt(Ftx2[:, 0]**2 + Ftx2[:, 1]**2)

        # Error total cuadrático (en píxeles²)
        error = d1**2 + d2**2
        # --------------------------------------

        # Inliers por umbral
        inliers = error < threshold
        votes = np.sum(inliers)

        # Actualizar el mejor modelo
        if votes > max_votes:
            max_votes = votes
            best_F = F
            best_inliers = inliers

    # Recalcular F con todos los inliers
    F_final = computeFundamental(x1[best_inliers], x2[best_inliers])
    return F_final, best_inliers



# ==========================================================
# Draw automatic epipolar lines using inlier matches
# ==========================================================

def compute_epipolar_lines_Diego(F, pts1, pts2, img1, img2, n_lines=15):
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


def compute_epipolar_lines(F, pts1, pts2, img1, img2, n_lines=15):
    """
    Draws epipolar lines in BOTH images:
      - in img2 for points from img1: l2 = F * x1
      - in img1 for points from img2: l1 = F^T * x2
    """
    img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    n = min(n_lines, len(pts1), len(pts2))
    if n == 0:
        return img1_c, img2_c
    idx = np.random.choice(len(pts1), size=n, replace=False)

    for i in idx:
        p1 = np.array([pts1[i,0], pts1[i,1], 1.0])
        p2 = np.array([pts2[i,0], pts2[i,1], 1.0])

        # Lines in image 2 from p1
        l2 = F @ p1
        a2,b2,c2 = l2 / (np.hypot(l2[0], l2[1]) + 1e-12)

        # Lines in image 1 from p2
        l1 = F.T @ p2
        a1,b1,c1 = l1 / (np.hypot(l1[0], l1[1]) + 1e-12)

        color = tuple(np.random.randint(0,255,3).tolist())

        # --- draw on img2 ---
        if abs(b2) > 1e-12:
            x0, x1p = 0, w2-1
            y0 = int((-c2 - a2*x0)/b2); y1 = int((-c2 - a2*x1p)/b2)
        else:  # vertical
            x0 = x1p = int(-c2/a2); y0, y1 = 0, h2-1
        cv2.line(img2_c, (int(x0), int(y0)), (int(x1p), int(y1)), color, 1)
        cv2.circle(img2_c, (int(p2[0]), int(p2[1])), 4, color, -1)

        # --- draw on img1 ---
        if abs(b1) > 1e-12:
            x0, x1p = 0, w1-1
            y0 = int((-c1 - a1*x0)/b1); y1 = int((-c1 - a1*x1p)/b1)
        else:
            x0 = x1p = int(-c1/a1); y0, y1 = 0, h1-1
        cv2.line(img1_c, (int(x0), int(y0)), (int(x1p), int(y1)), color, 1)
        cv2.circle(img1_c, (int(p1[0]), int(p1[1])), 5, color, -1)

    return img1_c, img2_c


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
    #base_path = "D:/Dev_Space/python/Lab3/"
    base_path = ""
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


def epipole(F, which='right'):
    # right epipole e2 is null of F^T; left epipole e1 is null of F
    import numpy as np
    if which == 'right':  # e2
        _,_,Vt = np.linalg.svd(F.T)
    else:                 # e1
        _,_,Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / (e[-1] + 1e-12)   # homogeneous -> pixel coords


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    # Choose between "sift" or "superglue"
    mode = "superglue"  # or "superglue"

    img1, img2, srcPts, dstPts = load_matches(mode)
    #F, inliers = ransacFundamental(srcPts, dstPts, iterations=10000, threshold=0.001)
    F, inliers = ransacFundamental(srcPts, dstPts, iterations=3000, threshold=8)
    print("Fundamental Matrix:\n", F)
    print(f"Inliers: {np.sum(inliers)} / {len(inliers)}")

    inlier_pts1 = srcPts[inliers]
    inlier_pts2 = dstPts[inliers]

    img1_epi, img2_epi = compute_epipolar_lines(
    F, inlier_pts1, inlier_pts2,
    img1, img2, len(inliers))

    e1 = epipole(F, 'left')   # image 1
    e2 = epipole(F, 'right')  # image 2
    print("e1:", e1[:2], "e2:", e2[:2], "(W,H) = ", img1.shape[1], img1.shape[0])

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


"""
Using SuperGlue correspondences, the RANSAC-estimated Fundamental Matrix produces consistent epipolar geometry. 
Epipolar lines in both images converge at a unique epipole, and the matched points lie close to their corresponding lines, 
confirming the correctness of the estimation. Compared to SIFT+NNDR, SuperGlue yields a higher number of inliers and cleaner geometry, 
demonstrating better robustness and match accuracy.
"""