import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# === Funciones auxiliares  ===
def ensamble_T(R_w_c, t_w_c):
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def multi_M(A,B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B.")
    return np.dot(A,B)

def invertir_T(T_w_c):
    R = T_w_c[0:3, 0:3]
    t = T_w_c[0:3, 3]
    T_c_w = np.eye(4)
    T_c_w[0:3, 0:3] = R.T
    T_c_w[0:3, 3] = -R.T @ t
    return T_c_w

def lineFromPoints(p1, p2):
    p1_h = np.array([p1[0], p1[1], 1])
    p2_h = np.array([p2[0], p2[1], 1])
    return np.cross(p1_h, p2_h)

def drawLine(l, strFormat, lWidth):
    p_l_y = np.hstack((0, -l[2] / l[1]))
    p_l_x = np.hstack((-l[2] / l[0], 0))
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)

# === MAIN ===
if __name__ == '__main__':
    R_w_c1 = np.loadtxt('D:/Dev_Space/python/Practice1/R_w_c1.txt')
    R_w_c2 = np.loadtxt('D:/Dev_Space/python/Practice1/R_w_c2.txt')
    t_w_c1 = np.loadtxt('D:/Dev_Space/python/Practice1/t_w_c1.txt')
    t_w_c2 = np.loadtxt('D:/Dev_Space/python/Practice1/t_w_c2.txt')
    K_c = np.loadtxt('D:/Dev_Space/python/Practice1/K.txt')

    T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
    T_w_c2 = ensamble_T(R_w_c2, t_w_c2)
    T_c1_w = invertir_T(T_w_c1)
    T_c2_w = invertir_T(T_w_c2)

    P_w_c1 = multi_M(K_c, T_c1_w[0:3, :])
    P_w_c2 = multi_M(K_c, T_c2_w[0:3, :])

    points = {
        "A": np.array([3.44, 0.80, 0.82]),
        "B": np.array([4.20, 0.80, 0.82]),
        "C": np.array([4.20, 0.60, 0.82]),
        "D": np.array([3.55, 0.60, 0.82]),
        "E": np.array([-0.01, 2.60, 1.21])
    }

    projected_points = {}
    for name, X in points.items():
        X_h = np.append(X, 1).reshape(4, 1)
        x1_h = P_w_c1 @ X_h
        x2_h = P_w_c2 @ X_h
        x1 = (x1_h[:2] / x1_h[2]).ravel()
        x2 = (x2_h[:2] / x2_h[2]).ravel()
        projected_points[name] = {"cam1": x1, "cam2": x2}

    # === Camera 1 ===
    img1 = cv.cvtColor(cv.imread("D:/Dev_Space/python/Practice1/Image1.jpg"), cv.COLOR_BGR2RGB)
    """Parentesis del punto 2.4 y  2.5"""

    A_3D = points["A"]      # Punto A en el mundo
    B_3D = points["B"]      # Punto B en el mundo
    d_AB = B_3D - A_3D      # Vector dirección (de A a B)
    AB_inf = np.array([*d_AB, 0]).reshape(4,1)  # Punto en el infinito (w=0)
    # Proyección del punto infinito
    ab_inf_cam1 = P_w_c1 @ AB_inf
    ab_inf_cam1 = ab_inf_cam1 / ab_inf_cam1[2]


    A_cam1, B_cam1 = projected_points["A"]["cam1"], projected_points["B"]["cam1"]
    C_cam1, D_cam1 = projected_points["C"]["cam1"], projected_points["D"]["cam1"]

    l_ab_cam1 = lineFromPoints(A_cam1, B_cam1)
    l_cd_cam1 = lineFromPoints(C_cam1, D_cam1)
    p12_cam1 = np.cross(l_ab_cam1, l_cd_cam1)
    p12_cam1 = p12_cam1 / p12_cam1[2]

    plt.figure(figsize=(10, 7))
    plt.imshow(img1)
    plt.title("Image 1: Lines l_AB, l_CD and intersection p₁₂")

    drawLine(l_ab_cam1, 'r-', 2)
    drawLine(l_cd_cam1, 'g-', 2)

    for name, p in zip(["A","B","C","D"], [A_cam1,B_cam1,C_cam1,D_cam1]):
        plt.plot(p[0], p[1], 'xr', markersize=8)
        plt.text(p[0]+10, p[1]-10, name, color='yellow', fontsize=10)

    plt.plot(ab_inf_cam1[0], ab_inf_cam1[1], 'oy', markersize=40)   
    plt.text(ab_inf_cam1[0]+30, ab_inf_cam1[1]-20, 'ab_inf', color='yellow', fontsize=12)
    plt.plot(p12_cam1[0], p12_cam1[1], 'ob', markersize=10)
    plt.text(p12_cam1[0]+15, p12_cam1[1]-10, 'p₁₂', color='blue', fontsize=12)
    plt.axis("image")
    plt.show()

    print("Cierra esta ventana para cargar la segunda imagen...")

    # === Camera 2 ===
    #Como ya tengo el punto al infinto en AB entonces solo debo pasarlo para aca
    # Proyección con la cámara 2
    ab_inf_cam2 = P_w_c2 @ AB_inf
    ab_inf_cam2 = ab_inf_cam2 / ab_inf_cam2[2]
    img2 = cv.cvtColor(cv.imread("D:/Dev_Space/python/Practice1/Image2.jpg"), cv.COLOR_BGR2RGB)

    A_cam2, B_cam2 = projected_points["A"]["cam2"], projected_points["B"]["cam2"]
    C_cam2, D_cam2 = projected_points["C"]["cam2"], projected_points["D"]["cam2"]

    l_ab_cam2 = lineFromPoints(A_cam2, B_cam2)
    l_cd_cam2 = lineFromPoints(C_cam2, D_cam2)
    p12_cam2 = np.cross(l_ab_cam2, l_cd_cam2)
    p12_cam2 = p12_cam2 / p12_cam2[2]

    plt.figure(figsize=(10, 7))
    plt.imshow(img2)
    plt.title("Image 2: Lines l_AB, l_CD and intersection p₁₂")

    drawLine(l_ab_cam2, 'y-', 2)
    drawLine(l_cd_cam2, 'b-', 2)

    for name, p in zip(["A","B","C","D"], [A_cam2,B_cam2,C_cam2,D_cam2]):
        plt.plot(p[0], p[1], 'xr', markersize=8)
        plt.text(p[0]+10, p[1]-10, name, color='yellow', fontsize=10)

    plt.plot(ab_inf_cam2[0], ab_inf_cam2[1], 'oy', markersize=40)
    plt.text(ab_inf_cam2[0]+15, ab_inf_cam2[1]-10, 'ab_inf', color='yellow', fontsize=12)
    plt.plot(p12_cam2[0], p12_cam2[1], 'ob', markersize=10)
    plt.text(p12_cam2[0]+15, p12_cam2[1]-10, 'p₁₂', color='blue', fontsize=12)
    plt.axis("image")
    plt.show()
