import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

basePath = "labSession5/"

def plot_camera(ax, T_wc, scale=0.05, label="cam"):
    """
    Plot a camera coordinate frame given T_wc (4x4: world <- camera).
    X axis: red, Y axis: green, Z axis: blue.
    """
    T_wc = np.asarray(T_wc, dtype=float).reshape(4, 4)
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]  # camera center in world frame

    # draw camera center
    ax.scatter(t[0], t[1], t[2], marker='o', color='k')
    ax.text(t[0], t[1], t[2], f" {label}", fontsize=10)

    # draw camera axes (X: red, Y: green, Z: blue)
    axis_length = scale
    axes_colors = ['r', 'g', 'b']
    axes_labels = ['x', 'y', 'z']

    for i in range(3):
        axis = R[:, i]  # column i of R is axis direction in world frame
        ax.quiver(
            t[0], t[1], t[2],
            axis[0], axis[1], axis[2],
            length=axis_length,
            color=axes_colors[i]
        )
        # Optional: label each axis end
        # ax.text(t[0] + axis[0]*axis_length,
        #         t[1] + axis[1]*axis_length,
        #         t[2] + axis[2]*axis_length,
        #         f"{label}_{axes_labels[i]}")


def set_equal_3d_axes(ax, X):
    """
    Set 3D axes to equal scale for a set of 3D points X (N,3).
    """
    X = np.asarray(X)
    x_min, y_min, z_min = X.min(axis=0)
    x_max, y_max, z_max = X.max(axis=0)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0:
        max_range = 1.0

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    half = 0.5 * max_range
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def main():
    # ------------------------------------------------------------
    # 1) Load 3D points and camera poses
    # ------------------------------------------------------------
    Xw = np.loadtxt("points3D_poseA.txt")   # (N,3)
    T_wc1 = np.loadtxt(basePath + "T_wc1.txt")        # (4,4)
    T_wc2 = np.loadtxt(basePath + "T_wc2.txt")        # (4,4)

    print("Loaded", Xw.shape[0], "3D points")

    # ------------------------------------------------------------
    # 2) Create 3D figure
    # ------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(Xw[:, 0], Xw[:, 1], Xw[:, 2], s=10, alpha=0.6)
    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_zlabel('Z (world)')
    ax.set_title('Pose A: Cameras and Triangulated Points')

    # Plot camera 1 and camera 2
    plot_camera(ax, T_wc1, scale=0.05, label="cam1")
    plot_camera(ax, T_wc2, scale=0.05, label="cam2")

    # Equal axis scale so geometry is not distorted
    set_equal_3d_axes(ax, np.vstack([Xw, T_wc1[:3, 3], T_wc2[:3, 3]]))

    plt.show()


if __name__ == "__main__":
    main()
