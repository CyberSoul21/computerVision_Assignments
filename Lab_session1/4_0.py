import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg

def point_to_plane_distance(point, plane):
    a, b, c, d = plane
    x, y, z = point
    return abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)


if __name__ == '__main__':

    A = np.array([3.44, 0.80, 0.82])
    B = np.array([4.20, 0.80, 0.82])
    C = np.array([4.20, 0.60, 0.82])
    D = np.array([3.55, 0.60, 0.82])

    X = np.vstack((np.array([A, B, C, D]).T, np.ones((1,4))))
    
    U, S, Vt = np.linalg.svd(X.T)
    pi = Vt[-1, :]
    pi = pi / np.linalg.norm(pi[:3])  # normalizar el vector normal
    print("π =", pi)

    for name, P in {"A":A, "B":B, "C":C, "D":D, "E":np.array([-0.01, 2.6, 1.21])}.items():
        dist = point_to_plane_distance(P, pi)
        print(f"d_{name} = {dist:.2f} m")