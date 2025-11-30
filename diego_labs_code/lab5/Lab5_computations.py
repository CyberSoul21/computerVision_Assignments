import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scOptim
import math

# ---------------------------------------------------------
# 1. Define base path for your Lab 5 files
# ---------------------------------------------------------
base_path = "D:/Dev_Space/python/Lab5/"

# ---------------------------------------------------------
# 2. Load intrinsic parameters (fisheye left/right)
# ---------------------------------------------------------
K_1 = np.loadtxt(base_path + "K_1.txt")    # 3x3 intrinsics cam1
K_2 = np.loadtxt(base_path + "K_2.txt")    # 3x3 intrinsics cam2

D1_k = np.loadtxt(base_path + "D1_k_array.txt")  # [k1 k2 k3 k4]
D2_k = np.loadtxt(base_path + "D2_k_array.txt")  # [k1 k2 k3 k4]

print("K1 shape:", K_1.shape)
print("K2 shape:", K_2.shape)
print("D1_k:", D1_k)
print("D2_k:", D2_k)

# ---------------------------------------------------------
# 3. Load extrinsics T_wc1, T_wc2  (world → camera)
# ---------------------------------------------------------
T_wc1 = np.loadtxt(base_path + "T_wc1.txt")   # 4x4
T_wc2 = np.loadtxt(base_path + "T_wc2.txt")   # 4x4

print("T_wc1 shape:", T_wc1.shape)
print("T_wc2 shape:", T_wc2.shape)

# ---------------------------------------------------------
# 4. (Optional) Invert to obtain camera → world
# ---------------------------------------------------------
T_c1w = np.linalg.inv(T_wc1)
T_c2w = np.linalg.inv(T_wc2)

R_wc1 = T_wc1[:3, :3]
t_wc1 = T_wc1[:3, 3]

R_wc2 = T_wc2[:3, :3]
t_wc2 = T_wc2[:3, 3]

# ---------------------------------------------------------
# 5. Load stereo baseline from left-to-right calibration
# ---------------------------------------------------------
T_leftRight = np.loadtxt(base_path + "T_leftRight.txt")  # optional

# ---------------------------------------------------------
# 6. Load motion (A → B)
# ---------------------------------------------------------
T_wAwB_seed = np.loadtxt(base_path + "T_wAwB_seed.txt")  # 4x4 seed
T_wAwB_gt   = np.loadtxt(base_path + "T_wAwB_gt.txt")    # 4x4 ground truth

# ---------------------------------------------------------
# 7. Load matched pixel points (u, v)
#     Format: 2 rows or 3 rows (u, v, 1)
# ---------------------------------------------------------
x1 = np.loadtxt(base_path + "x1.txt")   # frame A - cam1
x2 = np.loadtxt(base_path + "x2.txt")   # frame A - cam2
x3 = np.loadtxt(base_path + "x3.txt")   # frame B - cam1
x4 = np.loadtxt(base_path + "x4.txt")   # frame B - cam2

print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)
print("x3 shape:", x3.shape)
print("x4 shape:", x4.shape)

# Ensure points are 3xN homogeneous
def ensure_homogeneous(X):
    if X.shape[0] == 2:
        return np.vstack((X, np.ones((1, X.shape[1]))))
    return X

x1 = ensure_homogeneous(x1)
x2 = ensure_homogeneous(x2)
x3 = ensure_homogeneous(x3)
x4 = ensure_homogeneous(x4)

print("After homogenization:")
print("x1:", x1.shape, " x2:", x2.shape)
print("x3:", x3.shape, " x4:", x4.shape)

###Kannala Brandt implementation###
# - proyection and unproyection
#---------------------------------------------------------
# FishEye Proyection Function   
def proyection_3d(points, K, dCoeffiecients):
    #Parameters
    #fx, fy : focal lengths
    #cx, cy : principal point
    #dCoeffiecients: distortion coefficients [k1, k2, k3, k4]
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    k1, k2, k3, k4,_= dCoeffiecients
#---------------------------------------------------------
# Convert to spherical coordinates
    X, Y, Z = points.T
    esferic_xy = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(esferic_xy, Z)
    phi = np.arctan2(Y, X)
#|---------------------------------------------------------
# Apply distortion model
#---------------------------------------------------------
    theta2 = theta**2
    theta3 = theta* theta2
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    theta9 = theta7 * theta2

    #d_theta = k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9
    d_theta = theta+ (k1 * theta3 + 
                      k2 * theta5 + 
                      k3 * theta7 +
                      k4 * theta9)
#---------------------------------------------------------
    xd_norm = d_theta * np.cos(phi)
    yd_norm = d_theta * np.sin(phi)
    
    # Convert to pixel coordinates
    u = fx * xd_norm + cx
    v = fy * yd_norm + cy
    # Retornar como matriz 3xN (homogénea)
    fishes = np.vstack((u, v))
    ones = np.ones((1, fishes.shape[1]))
    return np.vstack((fishes, ones))

    
# Inverse Distorsion Function
#---------------------------------------------------------  
def inverse_distorsion(ray, dCoeffiecients):
    k1, k2, k3, k4,_ = dCoeffiecients
    theta = ray   # Valor inicial (el radio distorsionado r es una buena aproximación)
    coefs = [k4, 0, k3, 0, k2, 0, k1, 1, -ray ] # Coeficientes del polinomio
    roots = np.roots(coefs)
    # Filtrar raíces reales positivas
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-8)].real 
    for theta in real_roots:
        if theta >= 0 and theta <= np.pi: 
            valid_theta = theta
            break   
    if valid_theta is None:
        return ray
    return valid_theta
#---------------------------------------------------------
# FishEye unproyection Function
#---------------------------------------------------------
def unproyection (u_points, K , dCoeffiecients):
    """
    Unproyecta puntos de píxel (u, v) a un vector unitario (rayo) 3D 
    en coordenadas de cámara.
    
    Args:
        u_points (3xN numpy array): Puntos de imagen (u, v, 1).
        K (3x3 numpy array): Matriz intrínseca.
        dCoeffiecients (4-element array): Coeficientes de distorsión [k1, k2, k3, k4].
        
    Returns:
        3xN numpy array: Rayos unitarios (vx, vy, vz).
    """
    #Resulted rays
    rays = []
    #xc = k_inverse @ u_points
    k_inverse = np.linalg.inv(K)
 
    for i in range(u_points.shape[1]):
        # Extraer coordenadas de píxel (u, v, 1)
        #recorre cada columna hasta el final
        pixel_h = u_points[:, i]
        
        # 1. Convertir a coordenadas normalizadas y obtener (x_c, y_c)
        x_c, y_c, _ = k_inverse @ pixel_h
        
        # 2. Coordenadas Polares normalizadas
        #Z = 1?
        r = math.sqrt(x_c**2 + y_c**2) # Esto es d(θ)
        phi = math.atan2(y_c, x_c)


        # 3. Invertir distorsión para obtener el ángulo real (θ)
        theta = inverse_distorsion(r, dCoeffiecients)
        
        # 4. Construir el rayo unitario en coordenadas de cámara (vector v)
        # v = [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
        vx = math.sin(theta) * math.cos(phi)
        vy = math.sin(theta) * math.sin(phi)
        vz = math.cos(theta)
        
        rays.append([vx, vy, vz])
    
    # Retornar como matriz 3xN
    return np.array(rays).T

#Los rayos obtenidos seran usados para poder hacer los puntos de prueba
#---------------------------------------------------------
#3.2 triangulation implementation
#---------------------------------------------------------
def get_plane(v_ray): 
    """
    Obtiene el plano definido por un rayo unitario.
    
    Args:
        v_ray (3-element numpy array): Rayo unitario (vx, vy, vz).
        
    Returns:
        tuple: Dos planos (Pi_sym, Pi_perp) definidos por el rayo.
        """
    vx, vy, vz = v_ray
        # Pi_sym = [-vy, vx, 0, 0]^T
    Pi_sym = np.array([-vy, vx, 0, 0])
        # Pi_perp = [-vz*vx, -vz*vy, vx^2 + vy^2, 0]^T
    Pi_perp = np.array([-vz * vx, -vz * vy, vx**2 + vy**2, 0])
    return Pi_sym, Pi_perp

def triangulate_point_svd(v1_ray, v2_ray, T_wc1, T_wc2):
    """
    Triangula puntos 3D a partir de rayos unitarios de dos cámaras y sus transformaciones extrínsecas.
    
    Args:
        v1_ray (3xN numpy array): Rayos unitarios desde la cámara 1.
        v2_ray (3xN numpy array): Rayos unitarios desde la cámara 2.
        T_wc1 (4x4 numpy array): Transformación extrínseca de la cámara 1 (world → camera).
        T_wc2 (4x4 numpy array): Transformación extrínseca de la cámara 2 (world → camera).
        
    Returns:
        Nx3 numpy array: Puntos 3D triangulados en coordenadas del mundo.
    """
    pi_sym1, pi_perp1 = get_plane(v1_ray)
    pi_sym2, pi_perp2 = get_plane(v2_ray)

    #Transform planes to world coordinates
    pi_sym1_w = np.linalg.inv(T_wc1).T @ pi_sym1
    pi_perp1_w = np.linalg.inv(T_wc1).T @ pi_perp1
    pi_sym2_w = np.linalg.inv(T_wc2).T @ pi_sym2
    pi_perp2_w = np.linalg.inv(T_wc2).T @ pi_perp2

    #Build matrix A
    # A = [pi_sym1_w; pi_perp1_w; pi_sym2_w; pi_perp2_w] (4x4)
    A = np.vstack((pi_sym1_w, pi_perp1_w, pi_sym2_w, pi_perp2_w))
    U, S, Vh = np.linalg.svd(A)
    X_homogeneous = Vh[-1, :]  # Última fila de Vh
    return X_homogeneous



if __name__ == "__main__":
# --- Puntos de Prueba de Kannala-Brandt ---
# X1=[3, 2, 10], X2=[-5, 6, 7], X3=[1, 5, 14] (Nx3)
 X_test_c = np.array([
    [3.0, 2.0, 10.0], 
    [-5.0, 6.0, 7.0], 
    [1.0, 5.0, 14.0]])
# u1, u2, u3 (3xN)
U_expected = np.array([
    [503.387, 267.9465, 441.0609],  # <-- Coordenadas U
    [450.1594, 580.4671, 493.0671],  # <-- Coordenadas V
    [1.0,     1.0,      1.0]         # <-- Coordenadas Homogéneas
])

print("\n--- TEST DE VERIFICACIÓN KANNALA-BRANDT ---")

# --- TEST 1: PROYECCIÓN (3D -> 2D) ---
# points_c debe ser Nx3. Pasamos X_test_c.
#D1_k_real = D1_k[:4]  # Usamos los coeficientes cargados
u_pred = proyection_3d(X_test_c, K_1, D1_k) 

print ("--- Resultados de Proyección ---")
print (u_pred)
print ("--- Valores Esperados ---")
print (U_expected)

# --- TEST 2: UNPROYECCIÓN (2D -> Rayo 3D) ---
print("--- 2. Unproyección (2D Píxel -> Rayo Unitario) ---")

# 2a. Calcular Rayos Unitarios Esperados (Normalizando el punto 3D)
#We are normalizing the 3D points to get the expected unit rays.
#normalize implises dividing each point by its magnitude.
X_test_norms = np.linalg.norm(X_test_c, axis=1, keepdims=True)
rays_expected_normalized = X_test_c.T / X_test_norms.T
#The code gives the direction vector pointing from the origin to the point (3, 2, 10), (-5, 6, 7), and (1, 5, 14) respectively.
#This should match the output of your unproyection function.

print ("\nRayos Unitarios Esperados (Normalizados):")
print (np.round(rays_expected_normalized, 4))

# 2b. Calcular Rayos Obtenidos por la Unproyección
rays_pred = unproyection(U_expected, K_1, D1_k)
print ("\nRayos Unitarios Predichos (Tu Código):")
print (np.round(rays_pred, 4))

#--- TEST 3: TRIANGULACIÓN ---
#---------------------------------------------------------
print("\n--- 3.2 TRIANGULATION TEST ---")

rays_c1 = unproyection(x1, K_1, D1_k)  # Rayos desde cámara 1
rays_c2 = unproyection(x2, K_2, D2_k)  # Rayos desde cámara 2

N = rays_c1.shape[1]
points_3d = []

for i in range(N):
    v1_ray = rays_c1[:, i]
    v2_ray = rays_c2[:, i]
    X_homogeneous = triangulate_point_svd(v1_ray, v2_ray, T_wc1, T_wc2)
    # Convertir a coordenadas 3D dividiendo por la componente homogénea
    X_3d = X_homogeneous[:3] / X_homogeneous[3]
    points_3d.append(X_3d)

X_3D_reconstructed = np.array(points_3d).T 
print(f"Puntos 3D Reconstruidos (Matriz 3x{N}):\n")
print(X_3D_reconstructed)
print("\n--- End Triangulation ---")


# ---------------------------------------------------------
#This was made in ChatGPT
# Nombre del archivo de salida
file_name = "reconstructed_points_pose_A.txt"

# 1. Transponer la matriz de 3xN a Nx3 
# (Cada fila será un punto: [X, Y, Z], formato más común para visualización)
points_to_save = X_3D_reconstructed.T

# 2. Usar np.savetxt para guardar los datos
# fmt='%.6f': Asegura 6 decimales de precisión
# delimiter=' ': Usa espacio como separador entre X, Y, Z
np.savetxt(
    file_name, 
    points_to_save, 
    fmt='%.6f', 
    delimiter=' ', 
    comments='# '
)

print(f"\n Puntos 3D reconstruidos guardados exitosamente en: {file_name}")

   