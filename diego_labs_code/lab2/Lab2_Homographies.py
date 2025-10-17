import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ================================================================
# LAB 2 - SECTION 3.1: HOMOGRAPHY DEFINITION
# ================================================================

# -------------------------------
# Load Data
# -------------------------------
base_path = "D:/Dev_Space/python/Lab2/"

img1 = cv.cvtColor(cv.imread(base_path + "image1.png"), cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(cv.imread(base_path + "image2.png"), cv.COLOR_BGR2RGB)

T_w_c1 = np.loadtxt(base_path + "T_w_c1.txt")   # 4x4 pose: camera1 in world
T_w_c2 = np.loadtxt(base_path + "T_w_c2.txt")   # 4x4 pose: camera2 in world
K_c = np.loadtxt(base_path + "K_c.txt")           # 3x3 intrinsic matrix

# Plane equation in camera1 reference: π1 = [n_x, n_y, n_z, d]
plane_pi1 = np.array([0.0149, 0.9483, 0.3171, -1.7257])


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




# -------------------------------
# Compute Homography
# -------------------------------



def computeHomographyFromPose(poseCamera1world, poseCamera2world, K_instrinsict, plane):
    """
    Compute the homography relating image1 and image2 through a given plane
    defined in camera1 coordinates.
    H = K (R - t n^T / d) K^-1
    """
    # Relative pose between cameras
    T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1
    R = T_c2_c1[:3, :3]
    t = T_c2_c1[:3, 3].reshape(3, 1)

    # Plane parameters
    n = plane_pi1[:3].reshape(3, 1)
    d = plane_pi1[3]

    # Homography matrix
    H = K_instrinsict @ (R - (t @ n.T) / d) @ np.linalg.inv(K_instrinsict)
    H /= H[2, 2]  # normalize scale
    return H

#H_12 = computeHomographyFromPose(T_w_c1, T_w_c2, K_c, plane_pi1)

#print("Homography H_12 (image1 → image2):\n", H_12)

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


if __name__ == '__main__': 
   H_from_pose = computeHomographyFromPose(T_w_c1, T_w_c2, K_c, plane_pi1)
visualizeHomographyTransfer(img1, img2, H_from_pose)

# Run visualization


