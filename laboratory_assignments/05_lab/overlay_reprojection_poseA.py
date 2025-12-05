import numpy as np
import cv2
from kannalaBrandt import (projectKannalaBrandt, 
                            unprojectKannalaBrandt)

basePath = "labSession5/"

def overlay_points(image_path, x_pixels, Xw, K, D, T_wc, win_name):
    """
    image_path : path to the original fisheye image
    x_pixels   : (N,2) measured pixel coordinates (u,v)
    Xw         : (N,3) 3D points in WORLD frame
    K, D       : intrinsics and fisheye distortion
    T_wc       : (4,4) transform world<-camera (camera pose in world)
    win_name   : window name for display
    """
    # 1) Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load image {image_path}")
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 2) Transform 3D points into this camera frame: Xc = R_cw * Xw + t_cw
    T_wc = np.asarray(T_wc, dtype=float).reshape(4, 4)
    T_cw = np.linalg.inv(T_wc)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    Xc = (R_cw @ Xw.T + t_cw.reshape(3, 1)).T   # (N,3)

    # 3) Project with Kannala-Brandt
    x_proj = projectKannalaBrandt(Xc, K, D)    # (N,2)

    # 4) Draw original (green) and projected (red)
    for (u_meas, v_meas), (u_proj, v_proj) in zip(x_pixels, x_proj):
        # original measurement
        cv2.circle(img_color, (int(round(u_meas)), int(round(v_meas))),
                   4, (0, 255, 0), 1)  # green circle
        # reprojected point
        cv2.circle(img_color, (int(round(u_proj)), int(round(v_proj))),
                   2, (0, 0, 255), -1)  # red filled circle

    # 5) Show
    cv2.imshow(win_name, img_color)
    cv2.waitKey(0)
    #cv2.destroyWindow(win_name)


def main():
    # ------------------------------------------------------------
    # 1) Load 3D points (WORLD) and correspondences
    # ------------------------------------------------------------
    Xw = np.loadtxt("points3D_poseA.txt")   # (N,3)

    X1 = np.loadtxt(basePath + "x1.txt")  # (3,N)
    X2 = np.loadtxt(basePath + "x2.txt")  # (3,N)

    x1 = np.vstack((X1[0, :], X1[1, :])).T   # (N,2)
    x2 = np.vstack((X2[0, :], X2[1, :])).T   # (N,2)

    # ------------------------------------------------------------
    # 2) Load intrinsics + distortion
    # ------------------------------------------------------------
    K1 = np.loadtxt(basePath + "K_1.txt")
    K2 = np.loadtxt(basePath + "K_2.txt")
    D1_all = np.loadtxt(basePath + "D1_k_array.txt")
    D2_all = np.loadtxt(basePath + "D2_k_array.txt")
    D1 = D1_all[:4]
    D2 = D2_all[:4]

    # ------------------------------------------------------------
    # 3) Load camera poses
    # ------------------------------------------------------------
    T_wc1 = np.loadtxt(basePath + "T_wc1.txt")
    T_wc2 = np.loadtxt(basePath + "T_wc2.txt")

    # ------------------------------------------------------------
    # 4) Overlay for camera 1 and camera 2
    #    Adjust image filenames to your actual ones
    # ------------------------------------------------------------
    img1_path = basePath + "fisheye1_frameA.png"   # <-- change if different
    img2_path = basePath + "fisheye2_frameA.png"   # <-- change if different

    overlay_points(img1_path, x1, Xw, K1, D1, T_wc1, "Pose A - Cam1")
    #input()
    overlay_points(img2_path, x2, Xw, K2, D2, T_wc2, "Pose A - Cam2")


if __name__ == "__main__":
    main()
